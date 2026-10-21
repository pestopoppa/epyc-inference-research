#!/usr/bin/env python3
"""KPF-18 GPU gate runner: llama-server cross-slot prefix fork (--slot-fork-min-tokens), P1 of
handoffs/active/kv-prefix-fork-and-paged-attention.md (RTG-58). Written 2026-10-04; NOT yet run.

Subcommands (one per KPF-18 item, plus helpers):
  item1  equivalence vs fresh, DFlash2 on and off, against a fresh-vs-fresh control floor
  item2  fork-from-checkpoint vs the source's own continuation from that checkpoint (sequential, alone)
  item3  source invariance: source continuation with a child forked mid-decode vs the child at slot_fork false
  item4  mid-prefill checkpoints in mixed prefill+decode ubatches (prefill-budget trial-merge binary, -b/-ub 512)
  item5  seq-bit safety, cache_idle_slots on and off (/slots kv_pool invariant, used counts, memory-slot failures)
  item6  off = champion: fork build at --slot-fork-min-tokens 0 vs the champion binary, bit-identical
  item7  speed: 4-child fan-out over a 60k trunk, trunk-first, fork_off/fork_on in alternating windows A B A B
  all    items 1..7 in order
  build-prompt-set   launch the fork build once, tokenize the corpus via /tokenize, write the prompt set + sha256
  make-argv-file     render the gate argv file from the stack's :8083 launch record (no launch)

--dry-run validates inputs and prints the full plan with per-step estimated durations. It starts NO process
(no server, no linkage script, no SMI call, no claim): it only reads files, sysfs and /proc/net/tcp.

Process discipline: the runner starts llama-server as its own child (start_new_session) and signals ONLY that PID
(SIGTERM, wait, SIGKILL, then `ps -p` must fail). Nothing is ever signalled by name. Production :8083 is never
touched: if it is listening the runner refuses.

Records: one `epyc.kpf.run.v1` JSON per run (server launch x arm), shape per
/mnt/raid0/llm/tmp/kpf-p1-20261004/KPF03.md, written to <out>/records/<run_id>.json and appended to
<out>/belief_measurements.jsonl. stdlib only (`requests` is not installed on this host).
"""
from __future__ import annotations

# CPU ADAPTATION (2026-10-20, kv-prefix-fork P1 re-cut, branch experimental/kv-prefix-fork-p1r-20261020): `--cpu`
# selects a reduced CPU prompt-set spec (trunks 4k-16k, suffixes 256-1k, 16 pairs, speed group = 16k trunk) and
# skips every GPU-only check (KFD/VRAM sampler, :8083 refusal, HIP residency, mi210 claim, GPU window file).
# Everything else (records, equivalence math, read-back, linkage, own-PID process discipline) is unchanged.
# Copied from /mnt/raid0/llm/tmp/kpf-p1-20261004/kpf18/kpf18_gate.py.

import argparse
import glob
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpf18_arms as ARMS  # noqa: E402  KPF-18.1/.2/.3 arms: pure helpers + step builders (they take this module as `gate`)

HERE = Path(__file__).resolve().parent
SCHEMA = "epyc.kpf.run.v1"
PS_SCHEMA = "epyc.kpf18.promptset.v1"
LINKAGE = Path("/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh")
LAUNCH_RECORD_8083 = Path("/mnt/raid0/llm/epyc-orchestrator/logs/server_launches/8083.json")
WINDOW_FILE = Path("/mnt/raid0/llm/tmp/gpu-window/mi210.json")
CLAIM_MOD_DIR = Path("/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd")
CHAMPION_BIN = Path("/mnt/raid0/llm/kernels/builds/gpu-20260929-90c12df42/bin/llama-server")
FROZEN_TREE = Path("/mnt/raid0/llm/llama.cpp")
KFD_PROC = Path("/sys/class/kfd/kfd/proc")
DEFAULT_CARD = Path("/sys/class/drm/card2/device")  # the MI210 on this host
CORPUS_GLOB = "/mnt/raid0/llm/tmp/ds41-c95/contexts/C*/prompt.inline.txt"
DEFAULT_PREFIX = "numactl --membind=3 -- taskset -c 184-191"  # from the 8083 launch record
GIB = 1024 ** 3

# Production env. Source: the env recorded from live :8083 PID 1703677 (2026-10-04T02:48Z launch, same launch as
# logs/server_launches/8083.json) by /mnt/raid0/llm/tmp/gpu-block-27b-20261003/results/b4i/launch_ON.sh, as carried
# in /mnt/raid0/llm/tmp/kvu16h-vram-20261004/lib_kvu16h.py ENV_FALLBACK. LD_LIBRARY_PATH is rebuilt per binary.
PROD_ENV = {"HIP_PATH": "/opt/rocm", "ROCM_PATH": "/opt/rocm", "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores",
            "OMP_WAIT_POLICY": "active", "OMP_DYNAMIC": "false", "GGML_IQK": "1"}
LDP_TAIL = ["/opt/rocm/lib", "/usr/lib/llvm-20/lib", "/opt/AMD/aocc-compiler-5.0.0/lib"]
DROP_ENV_PREFIX = ("LLAMA_ARG_", "GGML_", "ROCBLAS_", "HIPBLASLT_", "AMD_LOG", "HIP_TRACE")
DROP_ENV_KEYS = ("LD_PRELOAD", "LLAMA_SERVER_SLOTS_DEBUG", "LD_LIBRARY_PATH")

# Gate production shape (handoff: ":8083 production argv" + the -b 512 -ub 512 + --no-cache-idle-slots package).
SHAPE_VALUES = [(("-np", "--parallel"), "4"), (("-c", "--ctx-size"), "393216"), (("-ctk", "--cache-type-k"), "q8_0"),
                (("-ctv", "--cache-type-v"), "q8_0"), (("-fa", "--flash-attn"), "on"), (("-ngl", "--n-gpu-layers"), "all"),
                (("--spec-type",), "draft-dflash"), (("--spec-draft-n-max",), "7"), (("--device", "-dev"), "ROCm0"),
                (("-b", "--batch-size"), "512"), (("-ub", "--ubatch-size"), "512"), (("--cache-ram", "-cram"), "65536")]
SHAPE_FLAGS = ["--kv-unified", "--no-mmap", "--no-cache-idle-slots"]
FORBIDDEN_IN_ARGV = ["--port", "--slot-fork-min-tokens", "--metrics", "--prefill-budget-decoding",
                     "--prefill-budget-target-ms"]

# Prompt-set spec (8 trunk groups x 4 suffixes = 32 pairs; group 5 is the 60k trunk item 7 fans out over).
TRUNKS = [8000, 16000, 24000, 32000, 48000, 60000, 72000, 80000]
SUFFIXES = [512, 2000, 4000, 8000]
SUFFIX_WORDS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
SRC_TAIL = 512        # items 2/3: source = trunk + this tail
SRC_TAIL_LONG = 8000  # item 4: source still prefilling for a while after its junction checkpoint
DECODER_LEN = 1024    # item 4: two decoding neighbours
SPEED_GROUP = 5       # 60k trunk
CPU_MODE = False
SPEC = {"min_pairs": 32, "trunk_range": (8000, 80000), "suffix_range": (500, 8000), "speed_T": 60000}


def set_cpu_spec() -> None:
    global TRUNKS, SUFFIXES, SRC_TAIL_LONG, SPEED_GROUP, CPU_MODE
    CPU_MODE = True
    TRUNKS = [4096, 8192, 12288, 16384]
    SUFFIXES = [256, 512, 768, 1024]
    SRC_TAIL_LONG = 4096
    SPEED_GROUP = 3
    SPEC.update(min_pairs=8, trunk_range=(4000, 16384), suffix_range=(256, 1024), speed_T=16384)

# ---------------------------------------------------------------- duration model (estimates only)
# Prefill ms/token = BASE + PER_K * occupied_k, where occupied_k = attention cells resident in the unified pool (own
# + foreign, thousands). Fit to the solo :8083-shape probe /mnt/raid0/llm/tmp/ds41-c95/probe-decode-vs-context.md
# (b10303, 2026-10-01: 851/807/600/487 tok/s at 2.5k/16k/50k/80k); the foreign-cell term is the KVU-16b mechanism
# (/mnt/raid0/llm/tmp/kvu16b-rootcause-20261004/REPORT.md section 1: without the masked-block skip a chunk reads every
# occupied cell). The fork build is on 90c12df42 (no skip), so the term applies. -b 512 costs ~10% solo prefill
# (same report, section 4 fix 1: "possibly 10-20% lower at shallow depth").
PF_BASE_MS, PF_PER_K_MS, B512_FACTOR = 1.15, 0.0235, 1.10
# Decode tok/s solo vs depth_k: DFlash2 n_max 7 ~37 -> ~30 (2k -> 80k), no-draft 30.8 -> 18.7 (same probe; Q38-T7
# run #1 shows DFlash2 ~= MTP at depth). 4 concurrent streams: aggregate = 1.3x the solo rate at the pool depth
# (assumption; Q38-T7 measured 20.84 tok/s aggregate for 4 x 16k WITH prefill interleave).
DEC_DF2 = (37.0, 0.09, 10.0)
DEC_NODRAFT = (30.8, 0.15, 8.0)
CONC_AGG = 1.3
LOAD_S, STOP_S, LINK_S, REQ_OVERHEAD_S = 90.0, 45.0, 5.0, 1.0  # :8083 log shows "listening" 9.2 s after start (page
# cache hot); 90 s covers a cold --no-mmap read of 27 GB + DFlash2 drafter + 393k-cell KV alloc.


def pf_s(n: int, occ: int, mixed: float = 1.0) -> float:
    if n <= 0:
        return 0.0
    ms = PF_BASE_MS + PF_PER_K_MS * (occ + n / 2) / 1000.0
    return n * ms / 1000.0 * B512_FACTOR * mixed


def dec_s(n: int, depth: int, dflash: bool = True, conc: int = 1) -> float:
    a, b, floor = DEC_DF2 if dflash else DEC_NODRAFT
    solo = max(floor, a - b * depth / 1000.0)
    per_stream = solo if conc <= 1 else solo * CONC_AGG / conc
    return n / per_stream


def fmt_s(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


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
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    """Replace (or append) a valued flag; value None removes it."""
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


def set_bool(argv: list[str], on: str, off: str, want_on: bool) -> list[str]:
    a = [t for t in argv if t not in (on, off)]
    return a + [on if want_on else off]


def port_listening(port: int) -> bool:
    """Read-only: /proc/net/tcp{,6} LISTEN rows. No subprocess."""
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for ln in Path(f).read_text().splitlines()[1:]:
                parts = ln.split()
                if len(parts) > 3 and parts[3] == "0A" and int(parts[1].rsplit(":", 1)[1], 16) == port:
                    return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def kfd_pids() -> dict[int, int]:
    res: dict[int, int] = {}
    try:
        for d in KFD_PROC.iterdir():
            if not d.name.isdigit():
                continue
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


def card_vram(card: Path) -> tuple[int | None, int | None]:
    try:
        return int((card / "mem_info_vram_used").read_text()), int((card / "mem_info_vram_total").read_text())
    except (OSError, ValueError):
        return None, None


def detect_smi() -> tuple[str | None, list[str] | None]:
    if shutil.which("rocm-smi"):
        return "rocm-smi", ["rocm-smi", "--showmeminfo", "vram", "--json"]
    if shutil.which("amd-smi"):
        return "amd-smi", ["amd-smi", "metric", "--mem-usage", "--json"]
    return None, None


def bytes_contain(path: Path, needle: bytes) -> bool | None:
    """Knob-compiled-in check without a subprocess (the `strings | grep` rule, CLAUDE.md Process Management)."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return needle in data


def window_status() -> dict:
    try:
        d = json.loads(WINDOW_FILE.read_text())
    except FileNotFoundError:
        return {"file": str(WINDOW_FILE), "exists": False}
    except (OSError, ValueError) as exc:
        return {"file": str(WINDOW_FILE), "exists": True, "error": str(exc)}
    return {"file": str(WINDOW_FILE), "exists": True, "holder": d.get("holder"), "parked_ports": d.get("parked_ports"),
            "parked_roles": d.get("parked_roles"), "expected_end": d.get("expected_end"),
            "preempt_requested_at": d.get("preempt_requested_at")}


# ---------------------------------------------------------------- equivalence
def top_probs(entry: dict) -> dict[int, float]:
    top = entry.get("top_logprobs") or entry.get("top_probs") or []
    out = {}
    for t in top:
        if "logprob" in t:
            out[int(t["id"])] = math.exp(float(t["logprob"]))
        elif "prob" in t:
            out[int(t["id"])] = float(t["prob"])
    return out


def _fill(d: dict[int, float], U: set) -> tuple[dict[int, float], float]:
    listed = sum(d.values())
    tail = max(0.0, 1.0 - listed)
    missing = [i for i in U if i not in d]
    kth = min(d.values()) if d else 0.0
    per = min(kth, tail / len(missing)) if missing else 0.0
    v = {i: d.get(i, per) for i in U}
    rest = max(1e-12, 1.0 - sum(v.values()))
    return v, rest


def compare(a: dict, b: dict, n_cmp: int) -> dict:
    """a, b: {"tokens": [...], "probs": [completion_probabilities entries]}."""
    ta, tb = a.get("tokens") or [], b.get("tokens") or []
    pref = 0
    for x, y in zip(ta[:n_cmp], tb[:n_cmp]):
        if x != y:
            break
        pref += 1
    pa = top_probs(a["probs"][0]) if a.get("probs") else {}
    pb = top_probs(b["probs"][0]) if b.get("probs") else {}
    U = set(pa) | set(pb)
    inter = set(pa) & set(pb)
    maxabs_u = max((abs(pa.get(i, 0.0) - pb.get(i, 0.0)) for i in U), default=None)
    maxabs_i = max((abs(pa[i] - pb[i]) for i in inter), default=None)
    kld = None
    if pa and pb:
        pv, pr = _fill(pa, U)
        qv, qr = _fill(pb, U)
        kld = sum(pv[i] * math.log(max(pv[i], 1e-12) / max(qv[i], 1e-12)) for i in U) + pr * math.log(pr / qr)
    identical = ta == tb and len(ta) > 0 and sha256_json(a.get("probs")) == sha256_json(b.get("probs"))
    return {"top1": bool(ta and tb and ta[0] == tb[0]), "prefix": pref, "n_cmp": min(n_cmp, max(len(ta), len(tb))),
            "maxabs_union": maxabs_u, "maxabs_intersection": maxabs_i, "kld": kld,
            "tokens_identical": ta == tb and len(ta) > 0, "identical": identical, "len_a": len(ta), "len_b": len(tb)}


def eq_summary(comps: list[dict], n_cmp: int) -> dict:
    if not comps:
        return {"top1_agree": 0, "n": 0, "logit_maxabs": 0.0, "kld": 0.0, "greedy_prefix_len_min": 0,
                "first_token_top1_agree": 0, "max_abs_logit": None, "max_kld": None,
                "identical_greedy_prefix_min": 0, "measured": False}
    mu = [c["maxabs_union"] for c in comps if c["maxabs_union"] is not None]
    mi = [c["maxabs_intersection"] for c in comps if c["maxabs_intersection"] is not None]
    kl = [c["kld"] for c in comps if c["kld"] is not None]
    pre = sorted(c["prefix"] for c in comps)
    return {"top1_agree": sum(c["top1"] for c in comps), "n": len(comps),
            "logit_maxabs": max(mu) if mu else None, "kld": max(kl) if kl else None,
            "greedy_prefix_len_min": pre[0], "measured": True,
            # VB-KVU-PF.1: the spec's names (README source-table row), beside the runner's legacy names
            "first_token_top1_agree": sum(c["top1"] for c in comps),
            "max_abs_logit": max(mu) if mu else None, "max_kld": max(kl) if kl else None,
            "identical_greedy_prefix_min": pre[0],
            "kld_mean": (sum(kl) / len(kl)) if kl else None,
            "prob_maxabs_intersection": max(mi) if mi else None,
            "greedy_prefix_full": sum(1 for c in comps if c["prefix"] >= n_cmp),
            "greedy_prefix_quartiles": [pre[0], pre[len(pre) // 4], pre[len(pre) // 2], pre[(3 * len(pre)) // 4], pre[-1]],
            "tokens_identical": sum(c["tokens_identical"] for c in comps),
            "bit_identical": sum(c["identical"] for c in comps)}


def ident_summary(comps: list[dict]) -> dict:
    return {"identical": sum(c["identical"] for c in comps), "n": len(comps),
            "tokens_identical": sum(c["tokens_identical"] for c in comps), "measured": bool(comps)}


EQ_BASIS = ("First-token top-k probabilities from /completion `n_probs` (completion_probabilities, post_sampling_probs "
            "false = softmax over the full vocabulary before sampling). The server cannot return full logits, so "
            "`logit_maxabs` is max |p_a - p_b| over the UNION of the two top-k sets (an id absent from one side counts "
            "as 0 there: an upper bound) and `kld` is KL(a||b) over the top-k union plus one tail bucket, absent ids "
            "imputed from that side's tail mass capped at its k-th prob (a coarse-grained KL: a lower bound on the "
            "full-vocab KL when the imputation is exact). Greedy-prefix length compares returned token ids over "
            "n_predict tokens (ignore_eos). `identical` = all token ids AND every per-position top-k entry equal.")


# ---------------------------------------------------------------- prompt set
class PromptSet:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.sha256: str | None = None
        self.groups: dict[int, dict] = {}
        self.decoders: list[list[int]] = []
        self.spec_only = True

    @classmethod
    def from_spec(cls) -> "PromptSet":
        ps = cls()
        for g, T in enumerate(TRUNKS):
            ps.groups[g] = {"T": T, "trunk": None, "suffixes": [None] * len(SUFFIXES), "suffix_lens": list(SUFFIXES),
                            "filler": None, "src_tail": None, "src_tail_long": None}
        return ps

    @classmethod
    def load(cls, path: Path) -> "PromptSet":
        ps = cls()
        ps.path, ps.spec_only = path, False
        ps.sha256 = sha256_file(path)
        side = path.with_name(path.name + ".sha256")
        if side.exists():
            want = side.read_text().split()[0]
            if want != ps.sha256:
                raise SystemExit(f"prompt set sha256 mismatch: {path} is {ps.sha256}, {side} says {want}")
        for ln in path.read_text().splitlines():
            r = json.loads(ln)
            k = r.get("kind")
            if k == "trunk":
                ps.groups.setdefault(r["group"], {"suffixes": {}, "suffix_lens": {}})
                ps.groups[r["group"]].update({"T": len(r["ids"]), "trunk": r["ids"]})
            elif k == "pair":
                g = ps.groups.setdefault(r["group"], {"suffixes": {}, "suffix_lens": {}})
                g["suffixes"][r["suffix_index"]] = r["suffix_ids"]
                g["suffix_lens"][r["suffix_index"]] = len(r["suffix_ids"])
            elif k in ("filler", "src_tail", "src_tail_long"):
                ps.groups.setdefault(r["group"], {"suffixes": {}, "suffix_lens": {}})[k] = r["ids"]
            elif k == "decoder":
                ps.decoders.append(r["ids"])
        for g in ps.groups.values():
            n = len(g["suffixes"])
            g["suffixes"] = [g["suffixes"][i] for i in range(n)]
            g["suffix_lens"] = [g["suffix_lens"][i] for i in range(n)]
        return ps

    def n_pairs(self) -> int:
        return sum(len(g["suffix_lens"]) for g in self.groups.values())

    def problems(self) -> list[str]:
        p = []
        if self.n_pairs() < SPEC["min_pairs"]:
            p.append(f"prompt set has {self.n_pairs()} pairs (< {SPEC['min_pairs']})")
        Ts = [g["T"] for g in self.groups.values()]
        lo, hi = SPEC["trunk_range"]
        if Ts and (min(Ts) < lo or max(Ts) > hi):
            p.append(f"trunk lengths {min(Ts)}..{max(Ts)} outside {lo}-{hi}")
        Ss = [s for g in self.groups.values() for s in g["suffix_lens"]]
        lo, hi = SPEC["suffix_range"]
        if Ss and (min(Ss) < lo or max(Ss) > hi):
            p.append(f"suffix lengths {min(Ss)}..{max(Ss)} outside {lo}-{hi}")
        if not self.spec_only:
            for gi, g in self.groups.items():
                firsts = [s[0] for s in g["suffixes"]]
                if len(set(firsts)) != len(firsts):
                    p.append(f"group {gi}: suffix first tokens not distinct {firsts}")
                for k in ("filler", "src_tail", "src_tail_long"):
                    if g.get(k) is None:
                        p.append(f"group {gi}: missing {k}")
            if len(self.decoders) < 2:
                p.append("prompt set needs 2 decoder prompts (item 4)")
        if SPEED_GROUP not in self.groups or abs(self.groups[SPEED_GROUP]["T"] - SPEC["speed_T"]) > 1000:
            p.append(f"group {SPEED_GROUP} must be the ~{SPEC['speed_T']} trunk (item 7)")
        return p


def corpus_texts(pattern: str) -> list[tuple[str, str]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no corpus files match {pattern}")
    return [(f, Path(f).read_text(errors="replace")) for f in files]


def rotated(corpus: list[tuple[str, str]], start: int, chars: int) -> str:
    parts, total, i = [], 0, start
    while total < chars:
        name, txt = corpus[i % len(corpus)]
        parts.append(f"\n\n===== {Path(name).parent.name} =====\n{txt}")
        total += len(parts[-1])
        i += 1
    return "".join(parts)[:chars]


def build_prompt_set(client: "Client", out_path: Path, corpus_glob: str, meta_extra: dict) -> Path:
    corpus = corpus_texts(corpus_glob)

    def ids_for(header: str, start: int, n: int) -> list[int]:
        chars = int(n * 4.2) + 2000
        for _ in range(6):
            toks = client.tokenize(header + rotated(corpus, start, chars))
            if len(toks) >= n:
                return toks[:n]
            chars = int(chars * 1.35 * n / max(1, len(toks)))
        raise RuntimeError(f"corpus too small for {n} tokens")

    lines = [{"kind": "meta", "schema": PS_SCHEMA, "trunks": TRUNKS, "suffixes": SUFFIXES, "src_tail": SRC_TAIL,
              "src_tail_long": SRC_TAIL_LONG, "decoder_len": DECODER_LEN,
              "corpus": [{"path": f, "sha256": hashlib.sha256(t.encode()).hexdigest(), "chars": len(t)} for f, t in corpus],
              **meta_extra}]
    nc = len(corpus)
    for g, T in enumerate(TRUNKS):
        lines.append({"kind": "trunk", "group": g, "ids": ids_for(f"[KPF-18 trunk g{g}]\n", g % nc, T)})
        for i, S in enumerate(SUFFIXES):
            sid = ids_for(f"{SUFFIX_WORDS[i]} section {g}.{i}:\n", (g + 3 + i) % nc, S)
            lines.append({"kind": "pair", "pair_id": g * len(SUFFIXES) + i, "group": g, "suffix_index": i,
                          "suffix_ids": sid})
        lines.append({"kind": "filler", "group": g, "ids": ids_for(f"[KPF-18 filler g{g}]\n", (g + 5) % nc, T)})
        lines.append({"kind": "src_tail", "group": g, "ids": ids_for(f"Source section {g}:\n", (g + 1) % nc, SRC_TAIL)})
        lines.append({"kind": "src_tail_long", "group": g,
                      "ids": ids_for(f"Source long section {g}:\n", (g + 2) % nc, SRC_TAIL_LONG)})
    for d in range(2):
        lines.append({"kind": "decoder", "index": d, "ids": ids_for(f"[KPF-18 decoder {d}]\n", (d + 4) % nc, DECODER_LEN)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in lines))
    sha = sha256_file(out_path)
    out_path.with_name(out_path.name + ".sha256").write_text(f"{sha}  {out_path.name}\n")
    ps = PromptSet.load(out_path)
    probs = ps.problems()
    if probs:
        raise SystemExit("built prompt set fails its own checks: " + "; ".join(probs))
    print(f"[kpf18] prompt set {out_path} sha256 {sha} ({ps.n_pairs()} pairs)", flush=True)
    return out_path


# ---------------------------------------------------------------- HTTP client
class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 3600.0):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            return e.code, e.read()[:2000].decode(errors="replace")
        ctype_json = raw[:1] in (b"{", b"[")
        return status, (json.loads(raw) if ctype_json else raw.decode(errors="replace"))

    def tokenize(self, text: str) -> list[int]:
        st, body = self._req("POST", "/tokenize", {"content": text, "add_special": False}, timeout=600)
        if st != 200:
            raise RuntimeError(f"/tokenize HTTP {st}: {str(body)[:200]}")
        return [t if isinstance(t, int) else t["id"] for t in body["tokens"]]

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

    def metrics(self) -> dict[str, float]:
        st, body = self._req("GET", "/metrics", timeout=60)
        out: dict[str, float] = {}
        if st == 200 and isinstance(body, str):
            for ln in body.splitlines():
                if ln.startswith("llamacpp:"):
                    k, _, v = ln.partition(" ")
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        return out

    def erase(self, slot: int) -> None:
        st, body = self._req("POST", f"/slots/{slot}?action=erase", {}, timeout=120)
        if st != 200:
            raise RuntimeError(f"erase slot {slot}: HTTP {st} {str(body)[:200]}")

    def complete(self, body: dict, timeout: float = 7200.0) -> dict:
        t0 = time.time()
        st, res = self._req("POST", "/completion", body, timeout=timeout)
        out = {"http": st, "wall_s": round(time.time() - t0, 3), "t_send": t0,
               "cache_prompt": body.get("cache_prompt")}
        if st != 200 or not isinstance(res, dict):
            out["error"] = str(res)[:800]
            return out
        out.update(tokens=res.get("tokens") or [], probs=res.get("completion_probabilities") or [],
                   timings=res.get("timings") or {}, timings_keys=sorted((res.get("timings") or {}).keys()))
        return out

    def stream(self, body: dict, stop: threading.Event | None = None, on_token=None, timeout: float = 7200.0) -> dict:
        body = {**body, "stream": True}
        req = urllib.request.Request(self.base + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        out = {"http": None, "t_send": t0, "ttft_ms": None, "tokens": [], "probs": [], "timings": {},
               "aborted": False, "token_times": []}
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out["http"] = r.status
                for raw in r:
                    if stop is not None and stop.is_set():
                        out["aborted"] = True
                        break
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    toks = ev.get("tokens") or []
                    if toks:
                        now = time.time()
                        if out["ttft_ms"] is None:
                            out["ttft_ms"] = round((now - t0) * 1000.0, 1)
                        out["tokens"].extend(toks)
                        out["token_times"].append((round(now, 3), len(toks)))
                        if on_token:
                            on_token(len(out["tokens"]))
                    out["probs"].extend(ev.get("completion_probabilities") or [])
                    if isinstance(ev.get("timings"), dict):
                        out["timings"] = ev["timings"]
        except urllib.error.HTTPError as e:
            out["http"] = e.code
            out["error"] = e.read()[:800].decode(errors="replace")
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"[:800]
        out["wall_s"] = round(time.time() - t0, 3)
        out["timings_keys"] = sorted(out["timings"].keys())
        return out


def body_for(ids: list[int], n_predict: int, slot: int | None, *, fork: bool | None = True, cache: bool = True,
             dflash: bool = True, ckpt: list[int] | None = None, n_probs: int = 20) -> dict:
    """fork=None: send no fork-specific field at all (champion-comparable body)."""
    b = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": cache, "return_tokens": True,
         "ignore_eos": True, "seed": 42}
    if n_probs:
        b["n_probs"] = n_probs
    if slot is not None:
        b["id_slot"] = slot
    if fork is False:
        b["slot_fork"] = False
    if ckpt:
        b["checkpoint_at"] = ckpt
    if not dflash:
        b["speculative.n_max"] = 0
    return b


# ---------------------------------------------------------------- VRAM sampler
class Sampler:
    """1 Hz KFD per-pid VRAM + card sysfs, SMI tool every `smi_period` s. Started BEFORE the launch, stopped after
    the PID is verified dead, so samples bracket the process lifetime (sample DURING, never after)."""

    def __init__(self, path: Path, card: Path, ceiling_gib: float, smi_cmd: list[str] | None, smi_period: float = 5.0):
        self.sink, self.card, self.ceiling = Jsonl(path), card, int(ceiling_gib * GIB)
        self.smi_cmd, self.smi_period = smi_cmd, smi_period
        self.own_pid: int | None = None
        self.abort = threading.Event()
        self.abort_reason: str | None = None
        self.n = self.n_own_nonzero = self.own_peak = self.card_peak = self.kfd_count_max = 0
        self.consec_own_zero = 0
        self.first_t = self.last_t = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._last_smi = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            used, total = card_vram(self.card)
            pids = kfd_pids()
            own = pids.get(self.own_pid) if self.own_pid else None
            rec = {"t": round(t, 3), "card_used_b": used, "card_total_b": total, "own_pid": self.own_pid,
                   "own_vram_b": own, "kfd": {str(k): v for k, v in pids.items()}}
            if self.smi_cmd and t - self._last_smi >= self.smi_period:
                self._last_smi = t
                try:
                    r = subprocess.run(self.smi_cmd, capture_output=True, text=True, timeout=10)
                    rec["smi"] = r.stdout[-2000:]
                except Exception as exc:  # noqa: BLE001
                    rec["smi_err"] = str(exc)[:200]
            self.sink.add(rec)
            self.n += 1
            self.first_t = self.first_t or t
            self.last_t = t
            self.kfd_count_max = max(self.kfd_count_max, len(pids))
            if used:
                self.card_peak = max(self.card_peak, used)
            if self.own_pid:
                if own:
                    self.n_own_nonzero += 1
                    self.own_peak = max(self.own_peak, own)
                    self.consec_own_zero = 0
                else:
                    self.consec_own_zero += 1
            if not self.abort.is_set():
                if own and own > self.ceiling:
                    self.abort_reason = f"own pid KFD VRAM {own / GIB:.2f} GiB > ceiling {self.ceiling / GIB:.1f}"
                elif used and used > self.ceiling:
                    self.abort_reason = f"card used {used / GIB:.2f} GiB > ceiling {self.ceiling / GIB:.1f}"
                if self.abort_reason:
                    self.abort.set()
                    self.sink.add({"t": round(t, 3), "abort": self.abort_reason})
                    print(f"[kpf18] ABORT: {self.abort_reason}", flush=True)
            self._stop.wait(1.0)

    def start(self) -> "Sampler":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=15)

    def summary(self) -> dict:
        return {"samples": self.n, "own_vram_nonzero_samples": self.n_own_nonzero,
                "own_vram_peak_gib": round(self.own_peak / GIB, 3), "card_used_peak_gib": round(self.card_peak / GIB, 3),
                "kfd_process_count_max": self.kfd_count_max,
                "first_sample_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.first_t)) if self.first_t else None,
                "last_sample_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_t)) if self.last_t else None,
                "abort": self.abort_reason, "samples_file": str(self.sink.path)}


class Poller:
    """/metrics (1 Hz) and optionally /slots (0.5 Hz) DURING a phase; INTERFACE.md: do not poll faster than ~1 Hz."""

    def __init__(self, client: Client, sink: Path, slots_too: bool) -> None:
        self.client, self.sink, self.slots_too = client, Jsonl(sink), slots_too
        self.peak_used = None
        self.peak_logical_slots = 0
        self.peak_logical_metric = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        k = 0
        while not self._stop.is_set():
            rec: dict = {"t": round(time.time(), 3)}
            try:
                m = self.client.metrics()
                rec["metrics"] = {x: v for x, v in m.items() if "kv_cells" in x or "fork" in x or "requests" in x}
                u = m.get("llamacpp:kv_cells_used")
                if u is not None:
                    self.peak_used = max(self.peak_used or 0, int(u))
                lg = m.get("llamacpp:kv_cells_logical")
                if lg is not None:
                    self.peak_logical_metric = max(self.peak_logical_metric or 0, int(lg))
                if self.slots_too and k % 2 == 0:
                    ss = self.client.slots()
                    s = sum(int(x.get("n_prompt_tokens") or 0) for x in ss)
                    rec["slots_sum_n_prompt_tokens"] = s
                    rec["slots_busy"] = sum(1 for x in ss if x.get("is_processing"))
                    self.peak_logical_slots = max(self.peak_logical_slots, s)
            except Exception as exc:  # noqa: BLE001
                rec["err"] = str(exc)[:200]
            self.sink.add(rec)
            k += 1
            self._stop.wait(1.0)

    def start(self) -> "Poller":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=30)


# ---------------------------------------------------------------- server lifecycle (own PID only)
LOG_RES = {
    "kv_unified": re.compile(r"kv_unified\s*=\s*'?(true|false)'?"),
    "n_ctx_slot": re.compile(r"n_slots = (\d+), n_ctx_slot = (\d+)"),
    "fork_enabled": re.compile(r"cross-slot prefix fork enabled: mode = (\w+), min tokens = (\d+), cache_idle_slots = (\d)"),
    "fork_disabled": re.compile(r"--slot-fork-min-tokens (\d+): cross-slot prefix fork disabled: (.*)"),
    "build": re.compile(r"build[ :=]+(\d+) \(([0-9a-f]+)\)"),
    "fork_line": re.compile(r"slot fork: src = (\d+) \((idle|busy)\), kind = (\w+), n_fork_tokens = (\d+)"),
    "mem_slot_fail": re.compile(r"failed to find a memory slot"),
    "progress": re.compile(r"id\s+(\d+) \| task (\d+) \| prompt processing, n_tokens =\s*(\d+)"),
    "budget_iter": re.compile(r"prefill-budget: iter \d+: decode_rows = (\d+) \(slots \d+\), prompt = (\d+)"),
}


class Server:
    def __init__(self, ctx: "Ctx", label: str, binary: Path, fork_min: int | None, *, cache_idle: bool | None = None,
                 extra_args: list[str] | None = None) -> None:
        self.ctx, self.label, self.binary, self.fork_min = ctx, label, Path(binary), fork_min
        self.cache_idle, self.extra_args = cache_idle, list(extra_args or [])
        self.bin_dir = self.binary.parent
        self.proc: subprocess.Popen | None = None
        self.log = ctx.out / "logs" / f"{label}.server.log"
        self.sampler: Sampler | None = None
        self.readback: dict = {}
        self.linkage: dict = {}
        self.stop_rec: dict = {}
        self.t_start = self.t_stop = None
        self.argv = self.build_argv()
        self.env, self.env_note = self.build_env()

    def build_argv(self) -> list[str]:
        a = list(self.ctx.gate_argv)
        a = set_flag(a, ("--slot-save-path",), str(self.ctx.out / "slot-save" / self.label))
        if self.cache_idle is not None:
            a = set_bool(a, "--cache-idle-slots", "--no-cache-idle-slots", self.cache_idle)
        a += ["--port", str(self.ctx.args.port), "--metrics"]
        if self.fork_min is not None:
            a += ["--slot-fork-min-tokens", str(self.fork_min)]
        a += self.extra_args
        return [str(self.binary)] + a

    def build_env(self) -> tuple[dict, dict]:
        drop = [k for k in os.environ if k.startswith(DROP_ENV_PREFIX) or k in DROP_ENV_KEYS]
        env = {k: v for k, v in os.environ.items() if k not in drop}
        env.update(PROD_ENV)
        env["LD_LIBRARY_PATH"] = ":".join([str(self.bin_dir)] + LDP_TAIL)
        return env, {"dropped_ambient": sorted(drop), "set": {**PROD_ENV, "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]}}

    def full_cmd(self) -> list[str]:
        return list(self.ctx.prefix) + self.argv

    def run_linkage(self) -> dict:
        t0 = time.time()
        r = subprocess.run(["bash", str(LINKAGE), str(self.binary), str(self.bin_dir)], capture_output=True, text=True,
                           env={k: v for k, v in self.env.items()}, timeout=120)
        rep = self.ctx.out / "logs" / f"{self.label}.linkage.txt"
        rep.parent.mkdir(parents=True, exist_ok=True)  # fix: the logs dir did not exist yet at the first launch
        rep.write_text(r.stdout + r.stderr)
        ok = r.returncode == 0 and any(ln.startswith("PASS") for ln in r.stdout.splitlines())
        return {"rc": r.returncode, "pass": ok, "report": str(rep), "s": round(time.time() - t0, 2),
                "cmd": f"LD_LIBRARY_PATH={self.env['LD_LIBRARY_PATH']} bash {LINKAGE} {self.binary} {self.bin_dir}"}

    def start(self) -> None:
        ctx = self.ctx
        if port_listening(ctx.args.port):
            raise Abort(f"port {ctx.args.port} already listening")
        if not CPU_MODE:
            if port_listening(8083):
                raise Abort("production :8083 is listening: its owner must stop it via orchestrator_stack.py first")
            foreign = {p: v for p, v in kfd_pids().items() if v > 0}
            if foreign:
                raise Abort(f"foreign KFD processes hold VRAM: {foreign}")
            used, _ = card_vram(ctx.card)
            if used is not None and used > ctx.args.vram_free_max_gib * GIB:
                raise Abort(f"card VRAM used {used / GIB:.2f} GiB before launch (> {ctx.args.vram_free_max_gib})")
        self.linkage = self.run_linkage()
        if not self.linkage["pass"]:
            raise Abort(f"verify_ggml_linkage.sh FAILED for {self.binary} (see {self.linkage['report']})")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        (ctx.out / "slot-save" / self.label).mkdir(parents=True, exist_ok=True)
        if not CPU_MODE:
            self.sampler = Sampler(ctx.out / "vram" / f"{self.label}.jsonl", ctx.card, ctx.args.vram_ceiling_gib,
                                   ctx.smi_cmd).start()
            time.sleep(2.0)  # >= 2 samples before the launch
        print(f"[kpf18] launch {self.label}: {shlex.join(self.full_cmd())}", flush=True)
        self.t_start = time.time()
        self.proc = subprocess.Popen(self.full_cmd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env,
                                     start_new_session=True)
        ctx.own_children.append(self)
        if self.sampler:
            self.sampler.own_pid = self.proc.pid
        threading.Thread(target=self._pump, daemon=True).start()
        t0 = time.time()
        client = Client(f"http://127.0.0.1:{ctx.args.port}")
        while time.time() - t0 < ctx.args.load_timeout_s:
            if self.proc.poll() is not None:
                raise Abort(f"server {self.label} exited rc={self.proc.returncode} before healthy (see {self.log})")
            if self.sampler and self.sampler.abort.is_set():
                raise Abort(f"VRAM ceiling during load: {self.sampler.abort_reason}")
            if client.health():
                break
            time.sleep(2.0)
        else:
            raise Abort(f"server {self.label} not healthy after {ctx.args.load_timeout_s} s")
        self.readback = self.read_back()
        ctx.client = client
        if not CPU_MODE:
            time.sleep(2.5)  # two 1 Hz samples after healthy, for the persistence check
            self.check_residency()

    def _pump(self) -> None:
        with open(self.log, "ab") as fh:
            for line in iter(self.proc.stdout.readline, b""):
                fh.write(b"%.3f " % time.time() + line)
                fh.flush()

    def log_text(self) -> str:
        try:
            return self.log.read_text(errors="replace")
        except OSError:
            return ""

    def read_back(self) -> dict:
        txt = self.log_text()
        rb: dict = {"pid": self.proc.pid}
        try:
            exe = os.path.realpath(f"/proc/{self.proc.pid}/exe")
            cmd = [x for x in Path(f"/proc/{self.proc.pid}/cmdline").read_bytes().decode().split("\0") if x]
        except OSError as exc:
            raise Abort(f"cannot read /proc for own pid {self.proc.pid}: {exc}")
        rb["exe"] = exe
        rb["exe_matches_binary"] = exe == os.path.realpath(self.binary)
        rb["proc_cmdline"] = cmd
        rb["proc_cmdline_matches_argv"] = cmd == self.argv
        m = LOG_RES["kv_unified"].search(txt)
        rb["kv_mode_from_launch_log"] = m.group(0) if m else None
        m = LOG_RES["n_ctx_slot"].search(txt)
        rb["n_slots_n_ctx_slot"] = [int(m.group(1)), int(m.group(2))] if m else None
        m = LOG_RES["fork_enabled"].search(txt)
        rb["slot_fork_from_launch_log"] = m.group(0) if m else None
        m2 = LOG_RES["fork_disabled"].search(txt)
        rb["slot_fork_disabled_line"] = m2.group(0) if m2 else None
        m = LOG_RES["build"].search(txt)
        rb["build_from_launch_log"] = f"{m.group(1)} ({m.group(2)})" if m else None
        try:
            maps = Path(f"/proc/{self.proc.pid}/maps").read_text()
        except OSError:
            maps = ""
        libs = sorted({ln.split()[-1] for ln in maps.splitlines() if ln.split() and ".so" in ln.split()[-1]
                       and ln.split()[-1].startswith("/")})
        ggml = [x for x in libs if "libggml" in x or "libllama" in x]
        rb["hip_mapped"] = [x for x in ggml if "libggml-hip" in x]
        rb["ggml_libs"] = ggml
        rb["ggml_all_from_bin_dir"] = bool(ggml) and all(os.path.realpath(x).startswith(str(self.bin_dir.resolve()) + "/")
                                                         for x in ggml)
        probs = []
        if not rb["exe_matches_binary"]:
            probs.append(f"/proc/{self.proc.pid}/exe = {exe}, not {self.binary}")
        if not rb["proc_cmdline_matches_argv"]:
            probs.append("/proc cmdline differs from the launched argv")
        if rb["kv_mode_from_launch_log"] is None or "true" not in rb["kv_mode_from_launch_log"]:
            probs.append(f"launch log kv_unified line: {rb['kv_mode_from_launch_log']!r} (need true)")
        if (not CPU_MODE and not rb["hip_mapped"]) or not rb["ggml_all_from_bin_dir"]:
            probs.append(f"HIP residency: libggml-hip mapped {rb['hip_mapped']}, all ggml from {self.bin_dir}: "
                         f"{rb['ggml_all_from_bin_dir']}")
        if self.fork_min:  # knob must have FIRED (positive control)
            if not rb["slot_fork_from_launch_log"]:
                probs.append(f"fork_on arm but no 'cross-slot prefix fork enabled' line ({rb['slot_fork_disabled_line']})")
            else:
                mm = LOG_RES["fork_enabled"].search(rb["slot_fork_from_launch_log"])
                if int(mm.group(2)) != self.fork_min:
                    probs.append(f"fork min tokens read back {mm.group(2)} != {self.fork_min}")
                if self.cache_idle is not None and int(mm.group(3)) != int(self.cache_idle):
                    probs.append(f"cache_idle_slots read back {mm.group(3)} != {int(self.cache_idle)}")
        elif rb["slot_fork_from_launch_log"]:
            probs.append("fork_off/champion arm but the fork-enabled line is present")
        rb["problems"] = probs
        if probs:
            raise Abort(f"launch read-back failed for {self.label}: " + "; ".join(probs))
        return rb

    def check_residency(self) -> None:
        s = self.sampler
        if s.consec_own_zero >= 2 or s.n_own_nonzero == 0:
            raise Abort(f"own pid {self.proc.pid} shows no KFD VRAM over {s.consec_own_zero} consecutive samples "
                        "(HIP residency not proven: CPU fallback?)")

    def observed_n_batch(self) -> int | None:
        """Modal per-iteration increment of 'prompt processing, n_tokens' for solo prefill: the effective n_batch."""
        last: dict[tuple, int] = {}
        deltas: dict[int, int] = {}
        for ln in self.log_text().splitlines():
            m = LOG_RES["progress"].search(ln)
            if m:
                key = (m.group(1), m.group(2))
                n = int(m.group(3))
                if key in last and n > last[key]:
                    d = n - last[key]
                    deltas[d] = deltas.get(d, 0) + 1
                last[key] = n
        return max(deltas, key=deltas.get) if deltas else None

    def log_counts(self) -> dict:
        txt = self.log_text()
        forks = LOG_RES["fork_line"].findall(txt)
        bi = LOG_RES["budget_iter"].findall(txt)
        return {"failed_to_find_memory_slot": len(LOG_RES["mem_slot_fail"].findall(txt)),
                "fork_lines": len(forks), "fork_lines_busy": sum(1 for f in forks if f[1] == "busy"),
                "fork_kinds": sorted({f[2] for f in forks}),
                "prefill_budget_iters": len(bi),
                "prefill_budget_mixed_iters": sum(1 for d, p in bi if int(d) > 0 and int(p) > 0),
                "n_batch_observed": self.observed_n_batch()}

    def stop(self) -> None:
        if self.proc is None:
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
        alive = subprocess.run(["ps", "-p", str(self.proc.pid)], capture_output=True).returncode == 0
        rec.update(returncode=self.proc.returncode, verified_dead=not alive)
        self.t_stop = time.time()
        self.stop_rec = rec
        time.sleep(2.0)  # samples after exit bracket the lifetime
        if self.sampler:
            self.sampler.stop()
        if self in self.ctx.own_children:
            self.ctx.own_children.remove(self)
        self.ctx.client = None
        print(f"[kpf18] stopped {self.label} pid {self.proc.pid}: {rec}", flush=True)
        if alive:
            raise RuntimeError(f"own child {self.proc.pid} still alive after TERM/KILL")


class Abort(RuntimeError):
    pass


# ---------------------------------------------------------------- run context, records
class Ctx:
    def __init__(self, args, ps: PromptSet, gate_argv: list[str], prefix: list[str], argv_meta: dict) -> None:
        self.args, self.ps, self.gate_argv, self.prefix, self.argv_meta = args, ps, gate_argv, prefix, argv_meta
        self.out = Path(args.out)
        self.client: Client | None = None
        self.own_children: list[Server] = []
        self.card = Path(args.card)
        self.smi_name, self.smi_cmd = detect_smi()
        self.host = socket.gethostname()
        self.t0 = time.time()
        self.run_n = 0
        self.preempted = False
        self.bins_sha: dict[str, str | None] = {}
        self.open_runs: list["Run"] = []

    def bin_sha(self, p: Path) -> str | None:
        k = str(p)
        if k not in self.bins_sha:
            self.bins_sha[k] = sha256_file(p)
        return self.bins_sha[k]

    def check_boundary(self) -> None:
        for s in self.own_children:
            if s.sampler and s.sampler.abort.is_set():
                raise Abort(f"VRAM: {s.sampler.abort_reason}")
            if s.proc and s.proc.poll() is not None:
                raise Abort(f"server {s.label} died rc={s.proc.returncode}")
        w = window_status() if not CPU_MODE else {}
        if w.get("preempt_requested_at"):
            self.preempted = True
            raise Abort(f"GPU window preempt requested at {w['preempt_requested_at']}: draining at this boundary")
        if (time.time() - self.t0) / 60 > self.args.budget_min:
            raise Abort(f"time budget {self.args.budget_min} min exhausted")


class Run:
    """One record = one server launch x arm (x DFlash2 setting for item 1)."""

    def __init__(self, ctx: Ctx, item: int, arm: str, srv: Server, dflash2: bool, window_id: str | None = None):
        ctx.run_n += 1
        self.ctx, self.item, self.arm, self.srv, self.dflash2 = ctx, item, arm, srv, dflash2
        self.run_id = f"kpf18-{utc_tag()}-{ctx.run_n}"
        self.window_id = window_id or f"{ctx.args.window_id}-item{item}"
        self.rows = Jsonl(ctx.out / "rows" / f"{self.run_id}.jsonl")
        self.start_utc = now_iso()
        self.eq: dict[str, list] = {}
        self.ident: dict[str, list] = {}
        self.ttft: list[float] = []
        self.prompt_n = self.cache_n = self.n_fork = 0
        self.checks: dict[str, list] = {}
        self.extra: dict = {}
        self.unique_cells_peak = None
        self.unique_cells_method = None
        self.decode = {"per_slot": [], "aggregate": None}
        # VB-KVU-PF.1: KPF-17.x/18.x fields. None/empty = NOT MEASURED in this run (never a fabricated zero).
        self.kpf_obs: list[dict] = []          # explicit per-request observations (see Run.obs)
        self.lanes: set[str] = set()
        self.cache_skip = {"requests_cache_prompt_false": 0, "cache_n_zero": 0}
        self.dec_n = 0
        self.dec_ms = 0.0
        self.dec_slots: list = []
        self.cells_samples: list[int] = []
        ctx.open_runs.append(self)

    def _argv_spec(self) -> dict:
        """Speculative shape of the LAUNCHED server, read from its argv: dflash2 only if --spec-type draft-dflash AND -md is a DFlash2 drafter."""
        argv = list(getattr(self.srv, "argv", None) or [])
        st = flag_val(argv, ("--spec-type",))
        md = flag_val(argv, ("-md", "--model-draft"))
        return {"spec_type": st, "draft_model": md,
                "dflash2_server": bool(st == "draft-dflash" and md and "dflash2" in md.lower())}

    def check(self, name: str, ok: bool, detail=None) -> bool:
        self.checks.setdefault(name, []).append({"ok": bool(ok), "detail": detail})
        return ok

    def row(self, role: str, r: dict, **kw) -> dict:
        t = r.get("timings") or {}
        self.prompt_n += int(t.get("prompt_n") or 0)
        self.cache_n += int(t.get("cache_n") or 0)
        self.n_fork += int(t.get("n_fork_tokens") or 0)
        if r.get("ttft_ms") is not None:
            self.ttft.append(r["ttft_ms"])
        # VB-KVU-PF.1: derive what every request already carries, so no item leaves these null
        self.dec_n += int(t.get("predicted_n") or 0)
        self.dec_ms += float(t.get("predicted_ms") or 0.0)
        if t.get("predicted_per_second") is not None:
            self.dec_slots.append(t.get("predicted_per_second"))
        self.lanes.add(kw.pop("lane", None) or "completion_raw")  # every runner request is raw /completion
        if r.get("cache_prompt") is False:
            self.cache_skip["requests_cache_prompt_false"] += 1
            if int(t.get("cache_n") or 0) == 0:
                self.cache_skip["cache_n_zero"] += 1
        try:
            u = self.ctx.client.metrics().get("llamacpp:kv_cells_used")
            if u is not None:
                self.cells_samples.append(int(u))
        except Exception:  # noqa: BLE001 - a sample miss is recorded as absence, never fabricated
            pass
        first = top_probs(r["probs"][0]) if r.get("probs") else {}
        self.rows.add({"t": round(time.time(), 3), "run_id": self.run_id, "role": role, **kw, "http": r.get("http"),
                       "error": r.get("error"), "wall_s": r.get("wall_s"), "ttft_ms": r.get("ttft_ms"),
                       "n_tokens": len(r.get("tokens") or []), "tokens": r.get("tokens"),
                       "probs_sha256": sha256_json(r.get("probs")), "first_topk": first,
                       "timings": {k: t.get(k) for k in ("prompt_n", "cache_n", "prompt_ms", "predicted_n",
                                                          "predicted_ms", "predicted_per_second", "draft_n",
                                                          "draft_n_accepted", "n_fork_tokens", "fork_src_slot",
                                                          "fork_src_kind")},
                       "timings_keys": r.get("timings_keys")})
        if r.get("http") != 200 or r.get("error"):
            raise Abort(f"{role}: HTTP {r.get('http')} {r.get('error')}")
        return r

    def obs(self, role: str, *, checkpoint_set=None, tgt_kv_hit=None, recurrent_ckpt_pos=None,
            drafter_kv_hit=None, n_fork_tokens=None, tokens_lost_to_missing_checkpoint=None,
            drafter_drop_kind=None, lane=None, pinned_count=None, **kw) -> dict:
        """KPF-17.x/18.x per-request observation (checkpoint set from /slots, per-component hit lengths, ...).
        Fields left None are recorded as not measured. The minimum is derived here, never supplied."""
        comps = {k: v for k, v in (("tgt_kv_hit", tgt_kv_hit), ("recurrent_ckpt_pos", recurrent_ckpt_pos),
                                   ("drafter_kv_hit", drafter_kv_hit)) if v is not None}
        o = {"role": role, "checkpoint_set": None if checkpoint_set is None else sorted(int(x) for x in checkpoint_set),
             "hit_components": comps or None, "hit_min": min(comps.values()) if comps else None,
             "n_fork_tokens": n_fork_tokens, "tokens_lost_to_missing_checkpoint": tokens_lost_to_missing_checkpoint,
             "drafter_drop_kind": drafter_drop_kind, "lane": lane, "pinned_count": pinned_count, **kw}
        if lane:
            self.lanes.add(lane)
        self.kpf_obs.append(o)
        return o

    def kpf_ext(self) -> dict:
        """The VB-KVU-PF.1 additive fields of epyc.kpf.run.v1 (ext_version 1)."""
        ob = self.kpf_obs
        def col(k):
            return [o[k] for o in ob if o.get(k) is not None]
        mins = [o["hit_min"] for o in ob if o.get("hit_min") is not None]
        lost = col("tokens_lost_to_missing_checkpoint")
        pins = col("pinned_count")
        return {
            "ext_version": 1,
            "checkpoint_set": [{"role": o["role"], "positions": o["checkpoint_set"]} for o in ob
                               if o.get("checkpoint_set") is not None] or None,
            "hit_components": [{"role": o["role"], **o["hit_components"], "min": o["hit_min"],
                                "n_fork_tokens": o.get("n_fork_tokens")} for o in ob
                               if o.get("hit_components")] or None,
            "hit_length_min": min(mins) if mins else None,
            "hit_tokens": self.n_fork,  # sum of per-request n_fork_tokens read from the slot timings
            "tokens_lost_to_missing_checkpoint": sum(lost) if lost else None,
            "drafter_drop_kind": sorted({str(x) for x in col("drafter_drop_kind")}) or None,
            "request_lane": sorted(self.lanes) or None,
            "cache_skip_control": dict(self.cache_skip),
            "pinned_count": {"max": max(pins), "samples": len(pins)} if pins else None,
        }

    def finish(self) -> dict:
        ctx, srv = self.ctx, self.srv
        if self in ctx.open_runs:
            ctx.open_runs.remove(self)
        n_cmp = ctx.args.n_predict
        if self.unique_cells_peak is None and self.cells_samples:
            self.unique_cells_peak = max(self.cells_samples)
            self.unique_cells_method = ("max llamacpp:kv_cells_used sampled once after each request (a lower bound "
                                        "of the true peak; item 7 polls at 1 Hz instead)")
        if self.decode.get("aggregate") is None and self.dec_ms > 0:
            self.decode = {"per_slot": self.decode.get("per_slot") or self.dec_slots,
                           "aggregate": round(self.dec_n / (self.dec_ms / 1000.0), 3),
                           "aggregate_method": "sum predicted_n / sum predicted_ms over this run's requests "
                                               "(token-weighted mean decode rate; requests are sequential, not a "
                                               "concurrent aggregate)"}
        eq = {
            "control_floor": eq_summary(self.eq.get("control_floor", []), n_cmp),
            "vs_fresh": eq_summary(self.eq.get("vs_fresh", []), n_cmp),
            "bit_exact_ckpt": ident_summary(self.ident.get("bit_exact_ckpt", [])),
            "source_invariance": ident_summary(self.ident.get("source_invariance", [])),
            "off_equals_champion": ident_summary(self.ident.get("off_equals_champion", [])),
        }
        for k, v in self.eq.items():
            if k not in eq:
                eq[k] = eq_summary(v, n_cmp)
        for k, v in self.ident.items():
            if k not in eq:
                eq[k] = ident_summary(v)
        rows_sha = sha256_file(self.rows.path)
        rec = {
            "schema": SCHEMA, "kpf_task": "KPF-18", "kpf18_item": self.item, "run_id": self.run_id, "arm": self.arm,
            "binary": str(srv.binary), "binary_realpath": os.path.realpath(srv.binary),
            "binary_sha256": ctx.bin_sha(srv.binary),
            "build_from_launch_log": srv.readback.get("build_from_launch_log"),
            "argv": srv.argv, "launch_prefix": list(ctx.prefix),
            "argv_readback": {"proc_cmdline_matches_argv": srv.readback.get("proc_cmdline_matches_argv"),
                              "exe_matches_binary": srv.readback.get("exe_matches_binary")},
            "argv_source": ctx.argv_meta,
            "env": srv.env_note,
            "kv_mode_from_launch_log": srv.readback.get("kv_mode_from_launch_log"),
            "slot_fork_from_launch_log": srv.readback.get("slot_fork_from_launch_log"),
            "n_slots_n_ctx_slot_from_launch_log": srv.readback.get("n_slots_n_ctx_slot"),
            "prompt_set": str(ctx.ps.path), "prompt_set_sha256": ctx.ps.sha256,
            "dflash2": bool(self.dflash2 and self._argv_spec()["dflash2_server"]),  # KPF-18.0b.1: derived from the launched argv, never constant
            "spec_type_from_argv": self._argv_spec()["spec_type"], "draft_model_from_argv": self._argv_spec()["draft_model"],
            "speculative_requested": bool(self.dflash2 and self._argv_spec()["spec_type"]),
            "equivalence": eq, "equivalence_basis": EQ_BASIS, "n_probs": ctx.args.n_probs, "n_predict": n_cmp,
            "ttft_ms": self.ttft, "prompt_tokens_processed": self.prompt_n, "cache_n": self.cache_n,
            "n_fork_tokens": self.n_fork, "unique_cells_peak": self.unique_cells_peak,
            "unique_cells_method": self.unique_cells_method, "decode_tok_s": self.decode,
            "window_id": self.window_id, "gpu_window_id": ctx.args.gpu_window_id,
            "start_utc": self.start_utc, "end_utc": now_iso(), "host": ctx.host,
            "checks": {k: {"ok": sum(1 for c in v if c["ok"]), "n": len(v),
                           "failed": [c["detail"] for c in v if not c["ok"]][:20]} for k, v in self.checks.items()},
            "server_log": str(srv.log), "server_log_counts": srv.log_counts(),
            "residency": {"linkage": srv.linkage, "hip_mapped": srv.readback.get("hip_mapped"),
                          "ggml_all_from_bin_dir": srv.readback.get("ggml_all_from_bin_dir"),
                          "smi_tool": ctx.smi_name, "vram": srv.sampler.summary() if srv.sampler else None,
                          "process_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(srv.t_start))
                          if srv.t_start else None,
                          "process_stop_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(srv.t_stop))
                          if srv.t_stop else None, "stop": srv.stop_rec},
            "rows": str(self.rows.path), "rows_sha256": rows_sha,
            "kpf_ext": self.kpf_ext(),
            "instrument_class": "serving", "category": "CANDIDATE",
            "grading_note": "n=1 per cell, no codified protocol cited: an OBSERVATION; projects under VB-KVU-PF and "
                            "grades Judged/Located through claim_tuple.grade(). Counts, not a pass word.",
            **self.extra,
        }
        rec["record_sha256"] = sha256_json(rec)
        write_json(ctx.out / "records" / f"{self.run_id}.json", rec)
        Jsonl(ctx.out / "belief_measurements.jsonl").add(rec)
        print(f"[kpf18] record {self.run_id} item {self.item} arm {self.arm}: "
              + json.dumps({k: (v.get("n"), v.get("top1_agree", v.get("identical"))) for k, v in eq.items()}),
              flush=True)
        return rec


# ---------------------------------------------------------------- steps
class Step:
    def __init__(self, desc: str, est_s: float, fn=None) -> None:
        self.desc, self.est_s, self.fn = desc, est_s, fn


def start_steps(ctx: Ctx, srv: Server) -> list[Step]:
    return [Step(f"[{srv.label}] linkage + launch + healthy + read-back: {shlex.join(srv.full_cmd())}",
                 LINK_S + LOAD_S, srv.start)]


def stop_step(srv: Server, before=None) -> Step:
    """SIGTERM own pid (verified dead), THEN finalize this server's records, so the VRAM samples bracket the whole
    process lifetime. `before` runs while the server is still up (e.g. item 6's cross-binary comparison)."""
    def fn():
        try:
            if before:
                before()
        finally:
            try:
                srv.stop()
            finally:
                for r in list(srv.ctx.open_runs):
                    if r.srv is srv:
                        r.finish()
    return Step(f"[{srv.label}] SIGTERM own pid, verify dead, then finalize its records", STOP_S, fn)


def erase_all(ctx: Ctx, slots=(0, 1, 2, 3)) -> None:
    for s in slots:
        ctx.client.erase(s)


def gsel(ctx: Ctx, spec: str | None, default: list[int]) -> list[int]:
    if not spec:
        return default
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def G(ctx: Ctx, g: int) -> dict:
    return ctx.ps.groups[g]


def cat(a, b):
    return (a or []) + (b or [])


# ---- item 1
def item1(ctx: Ctx) -> list[Step]:
    a = ctx.args
    srv = Server(ctx, f"item1-fork_on-{utc_tag()}", Path(a.fork_bin), a.fork_min)
    steps = start_steps(ctx, srv)
    state: dict = {}

    def mk_runs():
        state["on"] = Run(ctx, 1, "fork_on", srv, True)
        state["off"] = Run(ctx, 1, "fork_on_dflash2_off", srv, False)  # VB-KVU-PF.1: was mislabelled fork_on
        state["on"].extra["dflash2_off_method"] = state["off"].extra["dflash2_off_method"] = (
            "DFlash2-off rows re-use each slot's own state: the same prompt is re-sent to the same slot with "
            "cache_prompt true, slot_fork false, speculative.n_max 0; the pinned checkpoint_at [L-1] set on the "
            "DFlash2-on request is restored and 1 token is re-processed, so the fork-derived and fresh-derived KV/"
            "recurrent states are compared under no-draft decode with identical shapes")
    steps.append(Step("open item-1 records (DFlash2 on, DFlash2 off)", 0, mk_runs))
    for g in gsel(ctx, a.item1_groups, sorted(ctx.ps.groups)):
        T = G(ctx, g)["T"]

        def trunk_first(g=g):
            ctx.check_boundary()
            erase_all(ctx)
            r = ctx.client.complete(body_for(G(ctx, g)["trunk"], 0, 0, fork=False, ckpt=[-1], n_probs=0))
            state["on"].row("trunk_first", r, group=g)
        steps.append(Step(f"item1 g{g} erase all + trunk-first T={T} (slot 0, n_predict 0, checkpoint_at [-1])",
                          pf_s(T, 0) + REQ_OVERHEAD_S, trunk_first))
        for i, S in enumerate(G(ctx, g)["suffix_lens"]):
            L = T + S
            n = a.n_predict
            est = (pf_s(L, T) + pf_s(L, T + L + n) + pf_s(S, T + 2 * L + 2 * n)
                   + 3 * dec_s(n, T + L, True) + 3 * dec_s(n, T + L, False) + 6 * REQ_OVERHEAD_S + 3 * pf_s(1, 3 * L))

            def pair(g=g, i=i):
                ctx.check_boundary()
                grp = G(ctx, g)
                P = cat(grp["trunk"], grp["suffixes"][i])
                ck = [len(P) - 1]
                on, off = state["on"], state["off"]
                kw = dict(group=g, suffix=i, n_prompt=len(P))
                # fix (2026-10-20): the fork child goes FIRST. Sent after fresh A/B it forked their own L-1
                # checkpoints (LCP = the whole prompt) instead of the trunk, so it never measured a trunk fork.
                C = on.row("fork_child", ctx.client.complete(body_for(P, n, 1, fork=True, cache=True, ckpt=ck,
                                                                     n_probs=a.n_probs)), slot=1, **kw)
                A = on.row("fresh_A", ctx.client.complete(body_for(P, n, 2, fork=False, cache=False, ckpt=ck,
                                                                  n_probs=a.n_probs)), slot=2, **kw)
                on.check("fresh_cache_n_zero", A["timings"].get("cache_n") == 0, A["timings"].get("cache_n"))
                B = on.row("fresh_B", ctx.client.complete(body_for(P, n, 3, fork=False, cache=False, ckpt=ck,
                                                                  n_probs=a.n_probs)), slot=3, **kw)
                on.check("fresh_cache_n_zero", B["timings"].get("cache_n") == 0, B["timings"].get("cache_n"))
                on.check("fork_taken_n_fork_eq_T", C["timings"].get("n_fork_tokens") == len(grp["trunk"]),
                         [g, i, C["timings"].get("n_fork_tokens"), C["timings"].get("fork_src_kind")])
                on.eq.setdefault("control_floor", []).append(compare(A, B, n))
                on.eq.setdefault("vs_fresh", []).append(compare(C, A, n))
                on.eq.setdefault("vs_fresh_B", []).append(compare(C, B, n))
                res = {}
                for role, slot in (("fresh_A", 2), ("fresh_B", 3), ("fork_child", 1)):
                    r = off.row(role + "_nodraft", ctx.client.complete(
                        body_for(P, n, slot, fork=False, cache=True, dflash=False, n_probs=a.n_probs)), slot=slot, **kw)
                    off.check("nodraft_restored_L_minus_1", r["timings"].get("cache_n") == len(P) - 1,
                              [role, r["timings"].get("cache_n"), len(P) - 1])
                    res[role] = r
                off.eq.setdefault("control_floor", []).append(compare(res["fresh_A"], res["fresh_B"], n))
                off.eq.setdefault("vs_fresh", []).append(compare(res["fork_child"], res["fresh_A"], n))
                off.eq.setdefault("vs_fresh_B", []).append(compare(res["fork_child"], res["fresh_B"], n))
                erase_all(ctx, (1, 2, 3))
            steps.append(Step(f"item1 g{g}.{i} T={T} S={S}: fresh A (slot 2) + fresh B (slot 3) full prefill, fork "
                              f"child (slot 1), {n} tok each; DFlash2-off re-runs x3", est, pair))
    steps.append(stop_step(srv))
    return steps


# ---- item 2
def item2(ctx: Ctx) -> list[Step]:
    a = ctx.args
    srv = Server(ctx, f"item2-fork_on-{utc_tag()}", Path(a.fork_bin), a.fork_min)
    steps = start_steps(ctx, srv)
    state: dict = {}
    steps.append(Step("open item-2 record", 0, lambda: state.update(r=Run(ctx, 2, "fork_on", srv, True))))
    n = a.n_predict
    for g in gsel(ctx, a.item2_groups, sorted(ctx.ps.groups)):
        T = G(ctx, g)["T"]

        def source(g=g, T=T):  # fix: bind T now (late-binding closure sent the LAST group's T)
            ctx.check_boundary()
            erase_all(ctx)
            grp = G(ctx, g)
            r = ctx.client.complete(body_for(cat(grp["trunk"], grp["src_tail"]), 0, 0, fork=False, ckpt=[T], n_probs=0))
            state["r"].row("source", r, group=g)
            ck = [c for c in (ctx.client.slots()[0].get("checkpoints") or []) if c.get("n_tokens") == T]
            state["r"].check("source_has_pinned_junction", bool(ck and ck[0].get("pinned")), [g, ck])
        steps.append(Step(f"item2 g{g} erase all + source slot 0 = trunk+tail ({T}+{SRC_TAIL}), checkpoint_at [{T}]",
                          pf_s(T + SRC_TAIL, 0) + REQ_OVERHEAD_S, source))
        for i, S in enumerate(G(ctx, g)["suffix_lens"]):
            def pair(g=g, i=i, T=T):  # fix: bind T now
                ctx.check_boundary()
                grp = G(ctx, g)
                P = cat(grp["trunk"], grp["suffixes"][i])
                R = state["r"]
                ctx.client.erase(1)
                C = R.row("fork_child", ctx.client.complete(body_for(P, n, 1, fork=True, n_probs=a.n_probs)),
                          group=g, suffix=i, slot=1)
                R.check("fork_kind_checkpoint", C["timings"].get("fork_src_kind") == "checkpoint"
                        and C["timings"].get("n_fork_tokens") == T,
                        [g, i, C["timings"].get("fork_src_kind"), C["timings"].get("n_fork_tokens")])
                O = R.row("source_own", ctx.client.complete(body_for(P, n, 0, fork=False, cache=True, n_probs=a.n_probs)),
                          group=g, suffix=i, slot=0)
                R.check("source_restored_own_ckpt_T", O["timings"].get("cache_n") == T, [g, i, O["timings"].get("cache_n")])
                R.ident.setdefault("bit_exact_ckpt", []).append(compare(C, O, n))
            steps.append(Step(f"item2 g{g}.{i} S={S}: fork child (slot 1) from source ckpt {T}, then source's own "
                              f"continuation (slot 0) from the same ckpt; {n} tok each",
                              2 * pf_s(S, 2 * T + S) + 2 * dec_s(n, 2 * T, True) + 3 * REQ_OVERHEAD_S, pair))
    steps.append(stop_step(srv))
    return steps


# ---- item 3
def item3(ctx: Ctx) -> list[Step]:
    a = ctx.args
    srv = Server(ctx, f"item3-fork_on-{utc_tag()}", Path(a.fork_bin), a.fork_min)
    steps = start_steps(ctx, srv)
    state: dict = {}
    steps.append(Step("open item-3 record", 0, lambda: state.update(r=Run(ctx, 3, "fork_on", srv, True))))
    n = a.n_predict
    si = a.item3_suffix
    for g in gsel(ctx, a.item3_groups, sorted(ctx.ps.groups)):
        T = G(ctx, g)["T"]
        S = G(ctx, g)["suffix_lens"][si]

        def scenario(g=g, T=T, S=S):  # fix: bind T, S now
            ctx.check_boundary()
            grp = G(ctx, g)
            R = state["r"]
            src_ids = cat(grp["trunk"], grp["src_tail"])
            child_ids = cat(grp["trunk"], grp["suffixes"][si])
            outs = {}
            for arm in ("fork", "nofork", "nofork_repeat"):
                erase_all(ctx)
                if arm == "fork":  # equal-length filler so the source's cells sit at the same offset in every arm
                    R.row("seed_filler", ctx.client.complete(body_for(grp["filler"], 0, 2, fork=False, cache=False,
                                                                       n_probs=0)), group=g, arm=arm)
                else:  # the child's own trunk + pinned junction: it prefills only the suffix, like the fork
                    R.row("seed_child_trunk", ctx.client.complete(body_for(grp["trunk"], 0, 1, fork=False, ckpt=[-1],
                                                                           n_probs=0)), group=g, arm=arm)
                trig = threading.Event()
                res: dict = {}

                def src_thread():
                    res["src"] = ctx.client.stream(body_for(src_ids, n, 0, fork=False, ckpt=[T], n_probs=a.n_probs),
                                                   on_token=lambda k: k >= a.trigger_tokens and trig.set())
                    trig.set()
                th = threading.Thread(target=src_thread, daemon=True)
                th.start()
                trig.wait(timeout=3600)
                res["child"] = ctx.client.stream(body_for(child_ids, n, 1, fork=(arm == "fork"), n_probs=a.n_probs))
                th.join(timeout=7200)
                src = R.row("source", res["src"], group=g, arm=arm, slot=0)
                ch = R.row("child", res["child"], group=g, arm=arm, slot=1)
                R.check("source_still_decoding_at_child_send", bool(src.get("token_times")) and
                        src["token_times"][-1][0] > ch["t_send"], [g, arm])
                if arm == "fork":
                    R.check("child_forked_from_busy_source", ch["timings"].get("n_fork_tokens") == T
                            and ch["timings"].get("fork_src_slot") == 0, [g, ch["timings"].get("n_fork_tokens")])
                else:
                    R.check("child_own_ckpt_restore", ch["timings"].get("cache_n") == T
                            and not ch["timings"].get("n_fork_tokens"), [g, arm, ch["timings"].get("cache_n")])
                outs[arm] = (src, ch)
            R.ident.setdefault("source_invariance", []).append(compare(outs["fork"][0], outs["nofork"][0], n))
            R.ident.setdefault("source_invariance_control", []).append(
                compare(outs["nofork"][0], outs["nofork_repeat"][0], n))
            R.eq.setdefault("source_fork_vs_nofork", []).append(compare(outs["fork"][0], outs["nofork"][0], n))
            R.eq.setdefault("source_nofork_vs_repeat", []).append(compare(outs["nofork"][0], outs["nofork_repeat"][0], n))
        est = 3 * (pf_s(T, 0) + pf_s(T + SRC_TAIL, T) + pf_s(S, 2 * T) + dec_s(n, 2 * T, True, conc=2)
                   + 4 * REQ_OVERHEAD_S)
        steps.append(Step(f"item3 g{g} T={T} S={S}: arms fork / nofork / nofork_repeat; source (slot 0, ckpt [{T}]) "
                          f"decodes {n}, child (slot 1) launched at {a.trigger_tokens} source tokens", est, scenario))
    steps.append(stop_step(srv))
    return steps


# ---- item 4
def item4(ctx: Ctx) -> list[Step]:
    a = ctx.args
    extra = shlex.split(a.prefill_budget_args)
    srv = Server(ctx, f"item4-pbudget-fork_on-{utc_tag()}", Path(a.fork_bin_prefill_budget or "MISSING"), a.fork_min,
                 extra_args=extra)
    steps = start_steps(ctx, srv)
    state: dict = {}

    def open_run():
        state["r"] = Run(ctx, 4, "fork_on", srv, True)
        state["r"].extra["prefill_budget_args"] = extra
    steps.append(Step("open item-4 record", 0, open_run))
    n = a.n_predict
    si = a.item4_suffix
    for g in gsel(ctx, a.item4_groups, sorted(ctx.ps.groups)):
        T = G(ctx, g)["T"]
        S = G(ctx, g)["suffix_lens"][si]

        def scenario(g=g, T=T, S=S):  # fix: bind T, S now
            ctx.check_boundary()
            grp = G(ctx, g)
            R = state["r"]
            src_ids = cat(grp["trunk"], grp["src_tail_long"])
            child_ids = cat(grp["trunk"], grp["suffixes"][si])
            outs = {}
            for arm in ("fork", "fresh", "fresh_repeat"):
                erase_all(ctx)
                stop = threading.Event()
                dres: dict = {}
                started = [threading.Event(), threading.Event()]
                dths = []
                for d in range(2):
                    def dec(d=d):
                        dres[d] = ctx.client.stream(body_for(ctx.ps.decoders[d], 16384, 1 + d, fork=False, n_probs=0),
                                                    stop=stop, on_token=lambda k, d=d: k >= 8 and started[d].set())
                        started[d].set()
                    th = threading.Thread(target=dec, daemon=True)
                    th.start()
                    dths.append(th)
                for e in started:
                    e.wait(timeout=600)
                sres: dict = {}
                sth = threading.Thread(target=lambda: sres.update(src=ctx.client.complete(
                    body_for(src_ids, 0, 0, fork=False, ckpt=[T], n_probs=0))), daemon=True)
                sth.start()
                hit = None
                deadline = time.time() + 3600
                while time.time() < deadline and sth.is_alive():
                    s0 = ctx.client.slots()[0]
                    cks = [c for c in (s0.get("checkpoints") or []) if c.get("n_tokens") == T]
                    if cks and s0.get("is_processing") and int(s0.get("n_prompt_tokens") or 0) < len(src_ids):
                        hit = {"t": round(time.time(), 3), "n_prompt_tokens": s0.get("n_prompt_tokens"),
                               "ckpt": cks[0]}
                        break
                    time.sleep(1.0)  # INTERFACE.md section 4: ~1 Hz max
                R.check("child_sent_mid_prefill_after_junction", hit is not None, [g, arm])
                ch = ctx.client.stream(body_for(child_ids, n, 3, fork=(arm == "fork"), cache=(arm == "fork"),
                                                n_probs=a.n_probs))
                stop.set()
                sth.join(timeout=7200)
                for th in dths:
                    th.join(timeout=120)
                R.row("child", ch, group=g, arm=arm, slot=3, trigger=hit)
                if "src" in sres:
                    R.row("source", sres["src"], group=g, arm=arm, slot=0)
                if arm == "fork":
                    R.check("fork_from_busy_ckpt", ch["timings"].get("n_fork_tokens") == T
                            and ch["timings"].get("fork_src_kind") == "checkpoint",
                            [g, ch["timings"].get("n_fork_tokens"), ch["timings"].get("fork_src_kind")])
                else:
                    R.check("fresh_cache_n_zero", ch["timings"].get("cache_n") == 0, [g, arm, ch["timings"].get("cache_n")])
                outs[arm] = ch
            R.eq.setdefault("vs_fresh", []).append(compare(outs["fork"], outs["fresh"], n))
            R.eq.setdefault("control_floor", []).append(compare(outs["fresh"], outs["fresh_repeat"], n))
        est = 3 * (pf_s(T + SRC_TAIL_LONG, 2 * DECODER_LEN, mixed=1.6) + dec_s(n, 2 * T, True, conc=3)
                   + 20 + 4 * REQ_OVERHEAD_S) + 2 * pf_s(T + S, T, mixed=1.6) + pf_s(S, 2 * T, mixed=1.6)
        steps.append(Step(f"item4 g{g} T={T} S={S}: arms fork / fresh / fresh_repeat; 2 decoders (slots 1,2) + source "
                          f"(slot 0) trunk+{SRC_TAIL_LONG} ckpt [{T}], child (slot 3) sent once the junction exists "
                          f"while the source is still prefilling", est, scenario))
    steps.append(stop_step(srv))
    return steps


# ---- item 5
def item5(ctx: Ctx) -> list[Step]:
    a = ctx.args
    steps: list[Step] = []
    for cache_idle in (True, False):
        tag = "cis_on" if cache_idle else "cis_off"
        srv = Server(ctx, f"item5-{tag}-{utc_tag()}", Path(a.fork_bin), a.fork_min, cache_idle=cache_idle)
        steps += start_steps(ctx, srv)
        state: dict = {}

        def open_run(state=state, srv=srv, cache_idle=cache_idle):
            state["r"] = Run(ctx, 5, "fork_on", srv, True)
            state["r"].extra["cache_idle_slots"] = cache_idle
        steps.append(Step(f"open item-5 record ({tag})", 0, open_run))
        for g in gsel(ctx, a.item5_groups, [0, 2, 4, 7]):
            T = G(ctx, g)["T"]
            S1 = G(ctx, g)["suffix_lens"][1]

            def scenario(g=g, state=state, T=T):  # fix: bind T now
                ctx.check_boundary()
                grp = G(ctx, g)
                R = state["r"]
                P = cat(grp["trunk"], grp["suffixes"][1])

                def inv(where, expect_used=None):
                    ss = ctx.client.slots()
                    pool = ss[0].get("kv_pool") or {}
                    priv = sum((x.get("kv_cells") or {}).get("private", 0) for x in ss)
                    R.check("pool_invariant_used_eq_private_plus_shared",
                            pool.get("used") == priv + pool.get("shared", -1), [g, where, pool, priv])
                    R.check("one_snapshot_per_slots_call", all(x.get("kv_pool") == pool for x in ss), [g, where])
                    m = ctx.client.metrics().get("llamacpp:kv_cells_used")
                    R.check("metrics_used_eq_slots_used", m is not None and int(m) == pool.get("used"), [g, where, m])
                    if expect_used is not None:
                        R.check("pool_used_eq_expected", pool.get("used") == expect_used,
                                [g, where, pool.get("used"), expect_used])
                conts = {}
                for pas, filler_slot in (("A_source_intact", 2), ("B_source_reused", 0)):
                    erase_all(ctx)
                    inv(f"{pas}/erased", 0)
                    R.row("trunk_first", ctx.client.complete(body_for(grp["trunk"], 0, 0, fork=False, ckpt=[-1],
                                                                      n_probs=0)), group=g, pas=pas)
                    inv(f"{pas}/trunk", T)
                    c = R.row("child", ctx.client.complete(body_for(P, 0, 1, fork=True, n_probs=0)), group=g, pas=pas)
                    R.check("child_forked", c["timings"].get("n_fork_tokens") == T, [g, pas, c["timings"]])
                    inv(f"{pas}/child", len(P))
                    R.row("filler", ctx.client.complete(body_for(grp["filler"], 0, filler_slot, fork=False, cache=False,
                                                                 n_probs=0)), group=g, pas=pas, slot=filler_slot)
                    inv(f"{pas}/filler")
                    k = R.row("continuation", ctx.client.complete(body_for(P, 128, 1, fork=False, cache=True,
                                                                           n_probs=a.n_probs)), group=g, pas=pas)
                    inv(f"{pas}/continuation")
                    conts[pas] = k
                R.ident.setdefault("seq_bit_continuation", []).append(
                    compare(conts["A_source_intact"], conts["B_source_reused"], 128))
                erase_all(ctx)
                inv("final/erased", 0)
            est = 2 * (pf_s(T, 0) + pf_s(S1, T) + pf_s(T, T + S1) + pf_s(T + S1, 2 * T + S1) * 0.5
                       + dec_s(128, 2 * T, True) + 8 * REQ_OVERHEAD_S)
            steps.append(Step(f"item5 [{tag}] g{g} T={T}: passes A (filler in slot 2) / B (filler overwrites the "
                              f"source slot 0); invariant + used checks at every step; continuation 128 tok compared",
                              est, scenario))
        steps.append(stop_step(srv))
    return steps


# ---- item 6
def item6(ctx: Ctx) -> list[Step]:
    a = ctx.args
    steps: list[Step] = []
    groups = gsel(ctx, a.item6_groups, [0, 2, 4, 7])
    n = a.n_predict
    shared: dict = {"champion": {}, "fork_off": {}}
    for arm, binary, fm in (("champion", Path(a.champion_bin), None), ("fork_off", Path(a.fork_bin), 0)):
        srv = Server(ctx, f"item6-{arm}-{utc_tag()}", binary, fm)
        steps += start_steps(ctx, srv)
        state: dict = {}
        steps.append(Step(f"open item-6 record ({arm})", 0,
                          lambda state=state, srv=srv, arm=arm: state.update(r=Run(ctx, 6, arm, srv, True))))
        est = 0.0
        for g in groups:
            T = G(ctx, g)["T"]
            S0, S1 = G(ctx, g)["suffix_lens"][0], G(ctx, g)["suffix_lens"][1]
            est += pf_s(T + S0, 0) + pf_s(S1 + 64, T) + 2 * dec_s(n, T, True) + 3 * REQ_OVERHEAD_S

        def seq(arm=arm, state=state):
            ctx.check_boundary()
            R = state["r"]
            for g in groups:
                grp = G(ctx, g)
                erase_all(ctx)
                for i in (0, 1):  # i=1 re-uses slot 0's own cache through its checkpoints: both binaries must agree
                    P = cat(grp["trunk"], grp["suffixes"][i])
                    r = R.row("req", ctx.client.complete(body_for(P, n, 0, fork=None, n_probs=a.n_probs)),
                              group=g, suffix=i)
                    shared[arm][(g, i)] = r
            ss = ctx.client.slots()
            shared[arm]["slots_keys"] = sorted({k for s in ss for k in s.keys()})
            shared[arm]["metrics_keys"] = sorted(ctx.client.metrics().keys())
        steps.append(Step(f"item6 [{arm}] groups {groups}: per group erase, trunk+s0 then trunk+s1 on slot 0 "
                          f"(own-cache reuse), {n} tok, no fork-specific request fields", est, seq))

        def fin(arm=arm, state=state):
            if arm != "fork_off":
                return
            R = state["r"]
            for key, r in shared["fork_off"].items():
                if isinstance(key, tuple) and key in shared["champion"]:
                    c = shared["champion"][key]
                    R.ident.setdefault("off_equals_champion", []).append(compare(r, c, n))
                    R.check("timings_keys_equal", r.get("timings_keys") == c.get("timings_keys"),
                            [key, r.get("timings_keys"), c.get("timings_keys")])
            R.check("slots_keys_equal", shared["fork_off"].get("slots_keys") == shared["champion"].get("slots_keys"),
                    [shared["fork_off"].get("slots_keys"), shared["champion"].get("slots_keys")])
            R.check("metrics_keys_equal", shared["fork_off"].get("metrics_keys") == shared["champion"].get("metrics_keys"),
                    [shared["fork_off"].get("metrics_keys"), shared["champion"].get("metrics_keys")])
            R.extra["champion_binary"] = str(a.champion_bin)
            R.extra["champion_binary_sha256"] = ctx.bin_sha(Path(a.champion_bin))
        steps.append(stop_step(srv, before=fin))
    return steps


# ---- item 7
def item7(ctx: Ctx) -> list[Step]:
    a = ctx.args
    steps: list[Step] = []
    grp_i = SPEED_GROUP
    T = G(ctx, grp_i)["T"]
    Ss = G(ctx, grp_i)["suffix_lens"]
    n = a.item7_n_predict
    summary: dict = {"windows": []}
    for w, arm in enumerate(["fork_off", "fork_on", "fork_off", "fork_on"]):
        wid = f"{a.window_id}-item7-{'AB'[w % 2]}{w // 2 + 1}"
        fm = a.fork_min if arm == "fork_on" else 0
        srv = Server(ctx, f"item7-{wid}-{arm}-{utc_tag()}", Path(a.fork_bin), fm)
        steps += start_steps(ctx, srv)
        state: dict = {}

        def fan(arm=arm, wid=wid, srv=srv, state=state):
            ctx.check_boundary()
            R = Run(ctx, 7, arm, srv, True, window_id=wid)
            state["r"] = R
            grp = G(ctx, grp_i)
            erase_all(ctx)
            R.row("trunk_first", ctx.client.complete(body_for(grp["trunk"], 0, None, fork=None, ckpt=[-1], n_probs=0)))
            poll = Poller(ctx.client, ctx.out / "poll" / f"{R.run_id}.jsonl", slots_too=True).start()
            time.sleep(2.0)
            res: dict = {}
            t_rel = time.time()
            ths = []
            for i in range(len(Ss)):
                def child(i=i):
                    res[i] = ctx.client.stream(body_for(cat(grp["trunk"], grp["suffixes"][i]), n, None, fork=None,
                                                        n_probs=0))
                th = threading.Thread(target=child, daemon=True)
                th.start()
                ths.append(th)
            for th in ths:
                th.join(timeout=7200)
            time.sleep(2.0)
            poll.stop()
            first_tok, last_tok, per_slot, gen = [], [], [], 0
            for i in range(len(Ss)):
                r = R.row("child", res[i], suffix=i)
                t = r.get("timings") or {}
                per_slot.append(t.get("predicted_per_second"))
                gen += int(t.get("predicted_n") or 0)
                if r.get("token_times"):
                    first_tok.append(r["token_times"][0][0])
                    last_tok.append(r["token_times"][-1][0])
                R.check("fork_expected" if arm == "fork_on" else "no_fork_fields",
                        (arm == "fork_on") == ("n_fork_tokens" in t), [i, t.get("n_fork_tokens"), t.get("fork_src_kind")])
            span = (max(last_tok) - min(first_tok)) if first_tok else None
            R.decode = {"per_slot": per_slot, "aggregate": round(gen / span, 3) if span else None,
                        "aggregate_method": "sum predicted_n / (last child's last token - first child's first token)"}
            R.extra["child_ttft_ms_from_release"] = [round((res[i]["t_send"] - t_rel) * 1000 + (res[i]["ttft_ms"] or 0), 1)
                                                     for i in range(len(Ss))]
            R.extra["fanout_prefill_tokens"] = sum(int((res[i].get("timings") or {}).get("prompt_n") or 0)
                                                   for i in range(len(Ss)))
            R.extra["expected_prefill_if_shared"] = T + sum(Ss)
            # KPF-18.0b.1: ONE source for both arms (llamacpp:kv_cells_used polled at 1 Hz).  The /slots n_prompt_tokens sum is a different
            # quantity (logical, not unique cells) and is kept only as slots_sum_peak; an arm whose series is absent records None, never a substitute.
            R.unique_cells_peak = poll.peak_used
            R.unique_cells_method = ("max llamacpp:kv_cells_used over /metrics polled at 1 Hz during the fan-out (same source for fork_off and fork_on)"
                                     if poll.peak_used is not None else
                                     "llamacpp:kv_cells_used series ABSENT in this arm: unique cells not measured (no substitute source used)")
            R.extra["unique_cells_source"] = "llamacpp:kv_cells_used"
            R.extra["slots_sum_peak"] = poll.peak_logical_slots
            R.extra["kv_cells_logical_peak"] = poll.peak_logical_metric
            summary["windows"].append({"window_id": wid, "arm": arm, "run_id": R.run_id})
        est_on = pf_s(T, 0) + sum(pf_s(S, T, mixed=1.2) for S in Ss) + dec_s(n, T * 1.3, True, conc=4) + 10
        est_off = pf_s(T, 0) + pf_s(Ss[0], T) + sum(pf_s(T + S, T + 2 * T, mixed=1.0) for S in Ss[1:]) \
            + dec_s(n, 4 * T, True, conc=4) + 10
        steps.append(Step(f"item7 window {wid} [{arm}, --slot-fork-min-tokens {fm}]: trunk-first {T} then 4 children "
                          f"(suffixes {Ss}) released together, {n} tok each, /metrics 1 Hz + /slots 0.5 Hz",
                          est_on if arm == "fork_on" else est_off, fan))
        steps.append(stop_step(srv))

    def write_summary():
        write_json(ctx.out / "item7_summary.json", {**summary, "note": "pairs windows A1/B1/A2/B2; no verdict word"})
    steps.append(Step("write item7_summary.json", 0, write_summary))
    return steps


ITEMS = {1: item1, 2: item2, 3: item3, 4: item4, 5: item5, 6: item6, 7: item7}
ARMS.register(ITEMS, sys.modules[__name__])  # 181/182/183 = KPF-18.1/.2/.3; never part of `all`


# ---------------------------------------------------------------- argv file
def parse_argv_file(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    body = " ".join(ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))
    return shlex.split(body), comments


def argv_problems(argv: list[str], allow_drift: bool) -> tuple[list[str], list[str]]:
    hard, drift = [], []
    if argv and argv[0].endswith("llama-server"):
        hard.append("argv file must not contain the binary")
    for f in FORBIDDEN_IN_ARGV:
        if f in argv:
            hard.append(f"argv file must not contain {f} (the runner sets it)")
    for names, val in SHAPE_VALUES:
        got = flag_val(argv, names)
        if got != val:
            drift.append(f"{names[0]} = {got!r}, gate shape expects {val!r}")
    for f in SHAPE_FLAGS:
        if f not in argv:
            drift.append(f"missing {f}")
    md = flag_val(argv, ("-md", "--model-draft")) or ""
    if "DFlash2" not in md:
        drift.append(f"-md {md!r} is not the DFlash2 drafter")
    return hard, drift


def cmd_make_argv_file(args) -> int:
    rec = json.loads(LAUNCH_RECORD_8083.read_text())
    raw = rec["argv"]
    idx = next(i for i, t in enumerate(raw) if t.endswith("llama-server"))
    prefix, argv = raw[:idx], raw[idx + 1:]
    sha_ok = hashlib.sha256("\0".join(raw).encode()).hexdigest() == rec.get("argv_sha256")
    a = set_flag(argv, ("--port",), None)
    a = set_flag(a, ("-b", "--batch-size"), "512")
    a = set_flag(a, ("-ub", "--ubatch-size"), "512")
    a = set_bool(a, "--cache-idle-slots", "--no-cache-idle-slots", False)
    out = Path(args.argv_out)
    hdr = [
        "# KPF-18 gate argv (production :8083 argv minus binary/--port, plus the -b 512 -ub 512 + --no-cache-idle-slots",
        "# package named in kv-prefix-fork-and-paged-attention.md). The runner adds --port, --metrics,",
        "# --slot-fork-min-tokens and its own --slot-save-path; item 4 adds the prefill-budget args.",
        f"# source: {LAUNCH_RECORD_8083} (schema {rec.get('schema')}, launched_at {rec.get('launched_at')},",
        f"#   pid {rec.get('pid')}, launcher {' '.join(rec.get('launcher_argv') or [])}, stack_commit {rec.get('stack_commit')},",
        f"#   argv_sha256 {rec.get('argv_sha256')} (recomputed match: {sha_ok}))",
        f"# launch prefix in that record: {shlex.join(prefix)}",
        f"# production binary in that record: {raw[idx]} (NOT used here; the runner takes --fork-bin/--champion-bin)",
        "# changes vs the record: removed --port 8083; -ub 2048 -> -b 512 -ub 512; added --no-cache-idle-slots",
    ]
    out.write_text("\n".join(hdr) + "\n" + "\n".join(shlex.join(a[i:i + 2]) if a[i].startswith("-") and i + 1 < len(a)
                                                     and not a[i + 1].startswith("-") else shlex.quote(a[i])
                                                     for i in _argv_line_starts(a)) + "\n")
    print(f"wrote {out} (prefix {shlex.join(prefix)}; record argv_sha256 match {sha_ok})")
    return 0


def _argv_line_starts(a: list[str]) -> list[int]:
    out, i = [], 0
    while i < len(a):
        out.append(i)
        i += 2 if (a[i].startswith("-") and i + 1 < len(a) and not a[i + 1].startswith("-")) else 1
    return out


# ---------------------------------------------------------------- preflight + plan
def preflight(args, ps: PromptSet, gate_argv: list[str], items: list[int]) -> tuple[list[str], list[str]]:
    probs, notes = [], []
    if not (1024 <= args.port <= 65535) or 8000 <= args.port <= 8999:
        probs.append(f"--port {args.port} refused: never 8000-8999 (production/standard ports) or < 1024")
    if port_listening(args.port):
        probs.append(f"--port {args.port} is already listening")
    bins = []
    if any(i in (1, 2, 3, 5, 6, 7) + ARMS.ARM_ITEMS for i in items):
        bins.append(("--fork-bin", args.fork_bin, b"--slot-fork-min-tokens"))
    if 6 in items:
        bins.append(("--champion-bin", args.champion_bin, None))
    if 4 in items:
        bins.append(("--fork-bin-prefill-budget", args.fork_bin_prefill_budget, b"--prefill-budget-decoding"))
    for name, p, knob in bins:
        if not p:
            probs.append(f"{name} not given")
            continue
        pp = Path(p)
        if not (pp.is_file() and os.access(pp, os.X_OK)):
            probs.append(f"{name} {p}: not an executable file")
            continue
        if os.path.realpath(pp).startswith(str(FROZEN_TREE) + "/"):
            probs.append(f"{name} resolves inside the frozen production tree {FROZEN_TREE}: refused")
        common = pp.parent / "libllama-common.so"
        if knob:
            have = bytes_contain(common, knob)
            if have is not True:
                probs.append(f"{name}: {knob.decode()} not compiled into {common} ({have})")
            if name == "--fork-bin-prefill-budget" and bytes_contain(common, b"--slot-fork-min-tokens") is not True:
                probs.append(f"{name}: --slot-fork-min-tokens not compiled in (not the trial merge)")
        if name == "--champion-bin" and bytes_contain(common, b"--slot-fork-min-tokens"):
            probs.append(f"{name} carries --slot-fork-min-tokens: that is not the champion")
        notes.append(f"{name} {p} (bin dir {pp.parent}; LD_LIBRARY_PATH {':'.join([str(pp.parent)] + LDP_TAIL)})")
    if not LINKAGE.exists():
        probs.append(f"missing {LINKAGE}")
    m = flag_val(gate_argv, ("-m", "--model"))
    md = flag_val(gate_argv, ("-md", "--model-draft"))
    for f in (m, md, flag_val(gate_argv, ("--chat-template-file",))):
        if f and not Path(f).exists():
            probs.append(f"argv file references missing {f}")
    probs += ps.problems()
    if port_listening(8083):
        notes.append("production :8083 is LISTENING now: an execute run would refuse until its owner stops it")
    if CPU_MODE:
        notes.append("CPU mode: no KFD/VRAM sampler, no :8083 refusal, no HIP residency, no mi210 claim, no GPU window")
        return probs, notes
    foreign = {p: v for p, v in kfd_pids().items() if v > 0}
    used, total = card_vram(Path(args.card))
    notes.append(f"card {args.card}: VRAM used {used / GIB if used else used:.2f} GiB of "
                 f"{(total or 0) / GIB:.2f}; KFD processes holding VRAM now: {len(foreign)}"
                 if used is not None else f"card {args.card}: no mem_info_vram_used (wrong --card?)")
    if used is None:
        probs.append(f"--card {args.card} has no mem_info_vram_used")
    name, cmd = detect_smi()
    notes.append(f"VRAM sampler: KFD sysfs per-pid at 1 Hz + {name or 'NO SMI TOOL'} "
                 f"({shlex.join(cmd) if cmd else '-'}) every 5 s")
    if not name:
        probs.append("neither rocm-smi nor amd-smi on PATH")
    w = window_status()
    notes.append(f"GPU window file: {json.dumps(w)}")
    if not args.gpu_window_id:
        probs.append("--gpu-window-id / KPF18_GPU_WINDOW_ID not set")
    try:
        sys.path.insert(0, str(CLAIM_MOD_DIR))
        from autokernel.resource import device_claim as dc  # noqa: E402
        c = dc.inspect_device_claim("mi210_0")
        h = (c.get("claim") or {}).get("holder") or {}
        notes.append(f"device claim mi210_0 (advisory read): state {c.get('state')}, holder "
                     f"{h.get('label') or h.get('pid') or '-'}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"device claim module not inspectable: {type(exc).__name__}: {str(exc)[:120]}")
    return probs, notes


def build_ctx(args, items: list[int]) -> tuple[Ctx, list[str], list[str]]:
    gate_argv, comments = parse_argv_file(Path(args.argv_file))
    hard, drift = argv_problems(gate_argv, args.allow_shape_drift)
    if args.prompt_set and Path(args.prompt_set).exists():
        ps = PromptSet.load(Path(args.prompt_set))
    else:
        ps = PromptSet.from_spec()
        if args.prompt_set:
            ps.path = Path(args.prompt_set)
    prefix = shlex.split(args.launch_prefix)
    meta = {"argv_file": str(Path(args.argv_file).resolve()), "argv_file_sha256": sha256_file(Path(args.argv_file)),
            "argv_file_header": comments, "shape_drift": drift, "shape_drift_allowed": bool(args.allow_shape_drift)}
    ctx = Ctx(args, ps, gate_argv, prefix, meta)
    probs, notes = preflight(args, ps, gate_argv, items)
    a_probs, a_notes = ARMS.preflight_arms(ctx, items, sys.modules[__name__])
    probs, notes = probs + a_probs, notes + a_notes
    probs = hard + ([] if args.allow_shape_drift else [f"shape drift: {d}" for d in drift]) + probs
    if args.allow_shape_drift and drift:
        notes.append(f"shape drift ALLOWED and recorded: {drift}")
    if not ps.spec_only:
        notes.append(f"prompt set {ps.path} sha256 {ps.sha256}: {ps.n_pairs()} pairs")
    elif args.build_prompt_set:
        notes.append(f"prompt set will be BUILT at run start (first server's /tokenize) from {args.corpus} -> "
                     f"{ps.path or Path(args.out) / 'prompt_set.jsonl'}; plan uses the spec lengths")
    else:
        probs.append("need --prompt-set FILE (existing) or --build-prompt-set")
    return ctx, probs, notes


def print_plan(ctx: Ctx, items: list[int], steps_by_item: dict[int, list[Step]], probs: list[str], notes: list[str]):
    a = ctx.args
    print("=" * 100)
    print(f"KPF-18 gate plan ({'DRY RUN: nothing is launched' if a.dry_run else 'EXECUTE'})  host {ctx.host}  "
          f"{now_iso()}")
    print(f"  out {a.out} | port {a.port} | fork-min {a.fork_min} | n_predict {a.n_predict} | n_probs {a.n_probs} | "
          f"window-id {a.window_id} | gpu-window-id {a.gpu_window_id}")
    print(f"  launch prefix: {shlex.join(ctx.prefix)}")
    print(f"  gate argv ({a.argv_file}): {shlex.join(ctx.gate_argv)}")
    for n_ in notes:
        print(f"  - {n_}")
    print(f"  rate model: prefill ms/tok = ({PF_BASE_MS} + {PF_PER_K_MS} x occupied_k) x {B512_FACTOR}; decode DFlash2 "
          f"{DEC_DF2[0]}-{DEC_DF2[1]}/k tok/s, no-draft {DEC_NODRAFT[0]}-{DEC_NODRAFT[1]}/k; 4-stream agg x{CONC_AGG}; "
          f"load {LOAD_S:.0f}s, stop {STOP_S:.0f}s")
    total = 0.0
    for it in items:
        st = steps_by_item[it]
        sub = sum(s.est_s for s in st)
        total += sub
        print("-" * 100)
        print(f"ITEM {it}: {len(st)} steps, est {fmt_s(sub)}")
        for s in st:
            print(f"   {fmt_s(s.est_s):>8}  {s.desc}")
    print("=" * 100)
    print(f"TOTAL estimate: {fmt_s(total)} ({total / 60:.0f} min) for items {items}; budget --budget-min "
          f"{a.budget_min:.0f}")
    if probs:
        print(f"PREFLIGHT would REFUSE an execute run ({len(probs)} problems):")
        for p in probs:
            print(f"   ! {p}")
    else:
        print("PREFLIGHT: inputs PASS (live-only checks run at execute: linkage, read-back, residency, claim)")
    return total


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["item1", "item2", "item3", "item4", "item5", "item6", "item7", "all",
                                    "build-prompt-set", "make-argv-file", *ARMS.ARM_CMDS])
    ap.add_argument("--fork-bin")
    ap.add_argument("--champion-bin", default=str(CHAMPION_BIN))
    ap.add_argument("--fork-bin-prefill-budget")
    ap.add_argument("--prefill-budget-args", default="--prefill-budget-decoding 256")
    ap.add_argument("--argv-file")
    ap.add_argument("--argv-out", default=str(HERE / "argv.8083-gate.txt"))
    ap.add_argument("--launch-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--port", type=int, default=18083)
    ap.add_argument("--fork-min", type=int, default=1024)
    ap.add_argument("--prompt-set")
    ap.add_argument("--build-prompt-set", action="store_true")
    ap.add_argument("--corpus", default=CORPUS_GLOB)
    ap.add_argument("--out")
    ap.add_argument("--window-id", default=None)
    ap.add_argument("--gpu-window-id", default=os.environ.get("KPF18_GPU_WINDOW_ID"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--n-probs", type=int, default=20)
    ap.add_argument("--trigger-tokens", type=int, default=32)
    ap.add_argument("--item7-n-predict", type=int, default=256)
    for k in (1, 2, 3, 4, 5, 6):
        ap.add_argument(f"--item{k}-groups", default=None, help="comma list of trunk-group indices 0..7")
    ap.add_argument("--item3-suffix", type=int, default=2)
    ap.add_argument("--item4-suffix", type=int, default=1)
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--vram-ceiling-gib", type=float, default=62.5)
    ap.add_argument("--vram-free-max-gib", type=float, default=2.0)
    ap.add_argument("--load-timeout-s", type=float, default=900)
    ap.add_argument("--card", default=str(DEFAULT_CARD))
    ap.add_argument("--allow-shape-drift", action="store_true")
    ap.add_argument("--no-device-claim", default=None, metavar="REASON",
                    help="skip taking the mi210_0 device claim (reason is recorded); default: take it")
    ap.add_argument("--cpu", action="store_true", help="CPU adaptation: reduced spec, GPU-only checks skipped")
    ARMS.add_arguments(ap)
    args = ap.parse_args()
    if args.cpu:
        set_cpu_spec()
        args.no_device_claim = args.no_device_claim or "CPU mode: no GPU used"

    if args.cmd == "make-argv-file":
        return cmd_make_argv_file(args)
    if not args.argv_file or not args.out:
        ap.error("--argv-file and --out are required")
    args.window_id = args.window_id or f"kpf18-{utc_tag()}"
    if args.cmd in ARMS.ARM_CMDS:
        items = list(ARMS.ARM_CMDS[args.cmd])
    else:
        items = list(range(1, 8)) if args.cmd == "all" else ([] if args.cmd == "build-prompt-set" else [int(args.cmd[4:])])
    if args.cmd == "build-prompt-set":
        args.build_prompt_set = True
    if args.budget_min is None:
        args.budget_min = 1e9  # set from the estimate below
    ctx, probs, notes = build_ctx(args, items or [1])
    steps_by_item = {it: ITEMS[it](ctx) for it in items}
    total = sum(s.est_s for st in steps_by_item.values() for s in st)
    if args.budget_min >= 1e9:
        args.budget_min = max(30.0, 1.5 * total / 60)
    print_plan(ctx, items, steps_by_item, probs, notes)
    if args.dry_run:
        print("DRY RUN complete: no process was started, nothing was sent.")
        return 0
    if probs:
        print("REFUSED (preflight).")
        return 2

    Path(args.out).mkdir(parents=True, exist_ok=True)
    claim_cm = None
    stack: list = []

    def cleanup(*_a):
        for s in list(ctx.own_children):
            try:
                s.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf18] cleanup of own pid failed: {exc}", flush=True)

    def on_sig(signum, _f):
        cleanup()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    rc = 0
    try:
        if args.no_device_claim is None:
            sys.path.insert(0, str(CLAIM_MOD_DIR))
            from autokernel.resource import device_claim as dc  # noqa: E402
            journal = dc.ClaimJournal(Path(args.out) / "claim" / "device-claim-journal.jsonl")
            try:
                claim_cm = dc.acquire_device_claim("mi210_0", purpose=f"KPF-18 gate items {items} on :{args.port}",
                                                   campaign_id="KPF-18", journal=journal, holder_label="kpf18",
                                                   timeout_s=0, max_hold_s=args.budget_min * 60 + 1800)
            except dc.DeviceClaimError as exc:
                print(f"[kpf18] REFUSED: mi210_0 device claim not acquired: {exc}", flush=True)
                return 75
            claim = claim_cm.__enter__()
            stack.append(claim_cm)
            write_json(Path(args.out) / "claim" / "device-claim-receipt.json", claim.receipt().to_dict())
        # KPF-18.1/.2/.3: the decision rules are written HERE, before any server starts, and every arm record carries
        # the sha256 of this block (`predeclared_rules_sha256`)
        write_json(Path(args.out) / "run_meta.json", {"args": vars(args), "items": items, "estimate_s": total,
                                                       "predeclared_rules": ARMS.rules_for(items),
                                                       "predeclared_rules_sha256": ARMS.rules_sha256(items),
                                                       "started": now_iso(), "notes": notes,
                                                       "device_claim": args.no_device_claim or "mi210_0 held",
                                                       "region_lock": {"KPF18_NO_REGION_LOCK":
                                                                       os.environ.get("KPF18_NO_REGION_LOCK"),
                                                                       "KPF18_CPU_LIST": os.environ.get("KPF18_CPU_LIST",
                                                                                                        "184-191")}})
        if args.build_prompt_set and ctx.ps.spec_only:
            first = Server(ctx, f"promptset-tokenizer-{utc_tag()}", Path(args.fork_bin), None)
            first.start()
            try:
                pth = Path(args.prompt_set) if args.prompt_set else Path(args.out) / "prompt_set.jsonl"
                build_prompt_set(ctx.client, pth, args.corpus,
                                 {"tokenizer_binary_sha256": ctx.bin_sha(Path(args.fork_bin)),
                                  "model": flag_val(ctx.gate_argv, ("-m", "--model"))})
            finally:
                first.stop()
            ctx.ps = PromptSet.load(pth)
            steps_by_item = {it: ITEMS[it](ctx) for it in items}  # re-plan on the real prompt set
        for it in items:
            for st in steps_by_item[it]:
                t0 = time.time()
                print(f"[kpf18 {time.strftime('%H:%M:%SZ', time.gmtime())}] item {it}: {st.desc} "
                      f"(est {fmt_s(st.est_s)})", flush=True)
                if st.fn:
                    st.fn()
                print(f"[kpf18]   done in {fmt_s(time.time() - t0)}", flush=True)
    except Abort as exc:
        print(f"[kpf18] ABORT: {exc}", flush=True)
        rc = 5 if ctx.preempted else 3
    except Exception as exc:  # noqa: BLE001
        print(f"[kpf18] ERROR: {type(exc).__name__}: {exc}", flush=True)
        rc = 4
    finally:
        cleanup()
        for r in list(ctx.open_runs):  # an aborted item still leaves a (partial) record
            r.extra["partial"] = True
            try:
                r.finish()
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf18] could not finalize partial record {r.run_id}: {exc}", flush=True)
        for cm in reversed(stack):
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                print(f"[kpf18] claim release failed: {exc}", flush=True)
    print(f"[kpf18] exit rc={rc} (0 ok, 2 refused, 3 abort, 4 error, 5 preempted/drained, 75 claim held)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
