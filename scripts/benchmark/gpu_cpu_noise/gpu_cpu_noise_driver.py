#!/usr/bin/env python3
"""GPU -> CPU timing contamination experiment (bias + noise inflation), driver.

Pre-registration: epyc-root docs/design/gpu-cpu-noise-contribution-prereg-20261021.md
(its sha256 is recorded in RUN_RECORD.json before the first measured rep).

What it does (one session, owned GPU):
  * CPU workload W1 = the codified canonical llama-bench recipe (bench_canonical.sh,
    Qwen3.6-35B-A3B-MTP-Q8_0, pp512 + tg128, -r 3, full host 0-95 -t 96). Each rep is ONE
    bench_canonical invocation, which takes its OWN CPU region claim (cpu-list 0-95) for the
    step. The driver never takes an outer region claim around it (that would deadlock).
  * GPU conditions (the champion GPU build, Qwen3.8-27B-Q8_0 + DFlash2, the 2026-10-04 receipt argv):
      G0  GPU idle, no KFD process
      G1  27B server resident, idle             host threads 184-191 (node 3), membind 3
      G2  27B decode loop, 1 stream              host threads 184-191 (node 3), membind 3
      G3  27B prefill-heavy loop (~16k prompts)  host threads 184-191 (node 3), membind 3
      G4  = G2 but host threads 136-143 (node 1 = the MI210's own NUMA node), membind 1
      S8  GPU idle + 8 pure-ALU spinners on 184-191 (SMT-only mechanism control)
  * Randomized balanced blocks: each block = shuffle([G0, G0, G1, G2, G3, G4] (+S8 on odd
    blocks)); each entry is a RUN of `run_len` back-to-back reps of one condition.
  * Per rep: 1 Hz sysfs sampler (gpu_busy_percent = the rocm-smi "GPU use" source, VRAM, KFD
    pids, per-CPU busy on the host-thread sets), foreign-CPU accounting, positive controls.
    rocm-smi itself is called only BETWEEN reps (it is a python process and would load the
    bench cores during a rep).

Lifecycle: kills only PIDs it spawned (registry <session>/children.pids with start times),
TERM -> wait -> KILL -> verify /proc gone. On SIGTERM it stops at once, cleans, writes END.json.

Modes:  run | dry-run | stub (lifecycle test: stub bench + stub HTTP server, no inference)
        client (internal: the GPU load generator) | stub-server (internal)

--workload w2-q38fn (addendum <run-dir>/PREREG.ADDENDUM-W2-Q38FN.md): the CPU workload is ONE persistent
Qwen3.8-Flash-Next UD-IQ4_XS server on 0-95, launched with the production :8074 recipe from the v10 kernel store
under its OWN `region-lock run --cpu-list 0-95` claim (held for its lifetime; no --gpu-quiet: hold-once). A rep is
1 prefill request (unique seeded ~4.1k-token prompt) + 2 decode requests (fixed seeded prompt, 256 tokens). Blocks
are ABBA [G0, G2, G2, G0]. The rows, timeline and END.json keep the W1 shapes (plus `workload`), so the analyzer
reads both; it adds the predeclared W2 CO-RUN / W1-transfer decision.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.server
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

SCHEMA_REP = "epyc.gpu_cpu_noise.rep.v1"
SCHEMA_RUN = "epyc.gpu_cpu_noise.run_record.v1"

# ---------------------------------------------------------------- fixed recipe
CODE_ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BENCH_CANONICAL = os.path.join(CODE_ROOT, "scripts/benchmark/bench_canonical.sh")
VERIFY_LINKAGE = os.path.join(CODE_ROOT, "scripts/utils/verify_ggml_linkage.sh")
CPU_MODEL = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
BENCH_ARGS = ["-p", "512", "-n", "128", "-r", "3"]
GPU_BUILD = "/mnt/raid0/llm/kernels/builds/gpu-20261020-dd3d9721c"   # champion dd3d9721c, build 10345
GPU_EXPECT_BUILD = "b10345-dd3d9721c"
GPU_MODEL = "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"
GPU_DRAFT = "/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf"
GPU_TEMPLATE = "/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja"
GPU_PORT = 18961
GPU_DEVICE_LOCK = "/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock"
MI210_PCI_ID = "1002:740F"
PLACEMENTS = {  # name -> (host cpu list, membind node)
    "q3": ("184-191", "3"),   # SMT siblings of physical 88-95 (node 3; GPU is on node 1 -> remote)
    "n1": ("136-143", "1"),   # SMT siblings of physical 40-47 (node 1 = the MI210's NUMA node)
}
SPIN_CPUS = "184-191"
CONDITIONS = {
    # name: (server placement or None, client mode or None, spinners)
    "G0": (None, None, False),
    "G1": ("q3", None, False),
    "G2": ("q3", "decode", False),
    "G3": ("q3", "prefill", False),
    "G4": ("n1", "decode", False),
    "S8": (None, None, True),
}
BASE_BLOCK = ["G0", "G0", "G1", "G2", "G3", "G4"]
MIN_VRAM_BYTES = 20 * 2**30

# ---------------------------------------------------------------- W2-Q38FN (served whole-host CPU model)
# Pre-registration: <run-dir>/PREREG.ADDENDUM-W2-Q38FN.md. ONE persistent Qwen3.8-Flash-Next UD-IQ4_XS server on
# 0-95 with the PRODUCTION serving recipe: the argv/env orchestrator_stack.py builds for architect_general (:8074),
# read 2026-10-22 from build_server_command()/build_launch_env() + orchestration/derived/stack_priors.yaml.
# Only the port and the slot-save path differ.
WORKLOADS = ("w1", "w2-q38fn")
Q38FN_STORE_LINK = "/mnt/raid0/llm/kernels/production/cpu"          # production v10 kernel store (read-only)
Q38FN_BINDIR = "/mnt/raid0/llm/kernels/builds/cpu-20260921-ffc1bac82/bin"   # = stack prior runtime.binary_dir
Q38FN_EXPECT_COMMIT = "ffc1bac82"                                   # v10, build 10303
Q38FN_MODEL_DIR = "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"
Q38FN_SHARDS = [f"{Q38FN_MODEL_DIR}/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
Q38FN_MODEL = Q38FN_SHARDS[0]
Q38FN_MTP = "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
Q38FN_PORT = 18962
Q38FN_CPUS = "0-95"
Q38FN_PRE_EVICT_GIB = 40          # stack_topology.yaml architect_general.numa_pre_evict_gib
Q38FN_MAX_NODE_SHARE = 40         # canonical_recipe.CANONICAL_MAX_NODE_SHARE_PCT
Q38FN_RAM_NEED_GIB = 140          # addendum section 5
PROD_Q38FN_PORT = 8074            # production architect_general: refuse to run beside it
REGION_LOCK = "/workspace/repos/epyc-orchestrator/scripts/region-lock"
NUMA_EVICT = os.path.join(CODE_ROOT, "scripts/utils/numa_evict.py")
PLACEMENT_CHECK = os.path.join(CODE_ROOT, "scripts/utils/numa_placement_check.sh")
RAM_ADMIT = "/mnt/raid0/llm/tmp/compute-flags/ram_admit.sh"
W2_SEED = "gpu-cpu-noise-w2-q38fn-20261022"
W2_BLOCK = ["G0", "G2", "G2", "G0"]                       # ABBA, fixed order
W2_BLOCK_G3 = ["G0", "G2", "G3", "G3", "G2", "G0"]        # --w2-with-g3 (NOT in the registered session)
W2_PREFILL_WORDS = 3080           # ~4.1k tokens
W2_DECODE_PREAMBLE_WORDS = 400    # ~0.55k-token fixed decode prompt
W2_DECODE_N = 256
W2_DECODES_PER_REP = 2
W2_PROMPT_N_MIN_FRAC = 0.9        # prefill prompt_n >= 0.9 x warm-up median (no prompt-cache reuse)


def q38fn_env(base: dict | None = None) -> dict:
    """Production env for architect_general (build_launch_env + runtime requirements), with the store bindir
    PREPENDED to LD_LIBRARY_PATH (three ggml generations on this host; every launcher sets its own)."""
    env = dict(os.environ if base is None else base)
    for k in [k for k in env if k.startswith(("GGML_", "OMP_", "KMP_", "LLAMA_"))]:
        env.pop(k)
    env.update({"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "active",
                "OMP_DYNAMIC": "false", "KMP_BLOCKTIME": "10", "GGML_IQK": "1",
                "LD_LIBRARY_PATH": f"{Q38FN_BINDIR}:/usr/lib/llvm-20/lib"})
    return env


def q38fn_server_argv(port: int, slot_dir: str) -> list[str]:
    """taskset/numactl prefix + the production argv (orchestrator_stack.build_server_command, architect_general)."""
    return ["taskset", "-c", Q38FN_CPUS, "numactl", "--interleave=all", f"{Q38FN_BINDIR}/llama-server",
            "-m", Q38FN_MODEL, "--host", "127.0.0.1", "--port", str(port), "-np", "1", "-c", "262144",
            "-t", "96", "-ub", "2048", "--flash-attn", "on", "--jinja", "-ctk", "f16", "-ctv", "f16",
            "--mlock", "--no-mmap", "--cache-ram", "32768", "-md", Q38FN_MTP, "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "4", "--draft-p-min", "0.5", "--reasoning", "off",
            "--slot-save-path", slot_dir, "--device", "none", "--device-draft", "none"]


def q38fn_launch_argv(port: int, slot_dir: str, tag: str, env: dict) -> list[str]:
    """The server's OWN CPU claim on 0-95, held for the server's lifetime (region-lock is the parent; bash execs
    into env -> taskset -> numactl -> llama-server, so the server is region-lock's direct child).
    HOLD-ONCE: no --gpu-quiet here -- the session already holds gpu-quiet EXCLUSIVE, and a shared request from
    inside it would wait on itself. Nothing else in the session takes a CPU claim (no bench_canonical)."""
    keys = ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC", "KMP_BLOCKTIME", "GGML_IQK",
            "LD_LIBRARY_PATH")
    envs = " ".join(f"{k}={env[k]}" for k in keys)
    evict = f"/usr/bin/python3 -I {NUMA_EVICT} --target-gib {Q38FN_PRE_EVICT_GIB}"
    serve = " ".join(q38fn_server_argv(port, slot_dir))
    return [REGION_LOCK, "run", "--cpu-list", Q38FN_CPUS, "--role", "bench", "--tag", tag, "--timeout-s", "600",
            "--", "bash", "-c", f"{evict} || echo 'WARN pre-evict short'; exec env {envs} {serve}"]


def make_schedule_w2(blocks: int, run_len: int, with_g3: bool = False) -> list[dict]:
    pat = W2_BLOCK_G3 if with_g3 else W2_BLOCK
    return [{"block": b, "run": j, "cond": cond, "reps": [f"b{b:02d}-r{j}-{cond}-{k}" for k in range(run_len)]}
            for b in range(blocks) for j, cond in enumerate(pat)]


def w2_words(seed: str, key: str, n: int) -> str:
    h = hashlib.blake2b(f"{seed}\x1f{key}".encode(), digest_size=16).digest()
    rng = random.Random(int.from_bytes(h, "big"))
    return " ".join(rng.choice(WORDS) for _ in range(n))


def w2_prefill_body(seed: str, i: int) -> dict:
    return {"prompt": f"[{i}] " + w2_words(seed, f"prefill{i}", W2_PREFILL_WORDS) +
            "\nSummarize the text above in one line.", "n_predict": 1, "temperature": 0, "top_k": 1,
            "seed": 7, "cache_prompt": False}


def w2_decode_body(seed: str) -> dict:
    return {"prompt": w2_words(seed, "decode", W2_DECODE_PREAMBLE_WORDS) +
            "\nWrite a long, detailed technical essay about how compilers optimize loops.",
            "n_predict": W2_DECODE_N, "temperature": 0, "top_k": 1, "seed": 7, "ignore_eos": True,
            "cache_prompt": False}


def w2_controls(r: dict) -> list[str]:
    """W2 per-rep controls (addendum section 2/3): server identity, no prompt-cache reuse, full decode length."""
    why = []
    if not (r.get("cpu_server") or {}).get("ok_after_rep"):
        why.append("Q38FN server not alive or not owning its port after the rep")
    ref = r.get("prompt_n_ref")
    pn = [n for n in ((r.get("pp") or {}).get("prompt_n") or []) if n is not None]
    if ref and any(n < W2_PROMPT_N_MIN_FRAC * ref for n in pn):
        why.append(f"prefill prompt_n {pn} < {W2_PROMPT_N_MIN_FRAC} x ref {ref} (prompt-cache reuse?)")
    dn = (r.get("tg") or {}).get("predicted_n") or []
    if dn and any(n != W2_DECODE_N for n in dn):
        why.append(f"decode predicted_n {dn} != {W2_DECODE_N}")
    return why


def child_pids(pid: int) -> list[int]:
    out = []
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            try:
                out += [int(x) for x in open(f"/proc/{pid}/task/{t}/children").read().split()]
            except OSError:
                continue
    except OSError:
        pass
    return out


def proc_comm(pid: int) -> str | None:
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return None


def gpu_server_argv(placement: str, port: int) -> list[str]:
    cpus, node = PLACEMENTS[placement]
    b = os.path.join(GPU_BUILD, "bin")
    return ["taskset", "-c", cpus, "numactl", f"--membind={node}", f"{b}/llama-server",
            "-m", GPU_MODEL, "--host", "127.0.0.1", "--port", str(port), "-np", "4", "-c", "393216",
            "-t", "8", "-ub", "2048", "--flash-attn", "on", "--jinja", "-ctk", "q8_0", "-ctv", "q8_0",
            "--kv-unified", "--no-mmap", "-ngl", "all", "--cache-ram", "65536",
            "--chat-template-file", GPU_TEMPLATE, "-md", GPU_DRAFT, "-ngld", "99",
            "--spec-type", "draft-dflash", "--spec-draft-n-max", "7", "--device", "ROCm0",
            "--device-draft", "ROCm0", "--slots", "--no-webui"]


# ---------------------------------------------------------------- small utils
def utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def self_hash(row: dict) -> dict:
    body = json.dumps({k: v for k, v in row.items() if k != "content_sha256"}, sort_keys=True).encode()
    row["content_sha256"] = hashlib.sha256(body).hexdigest()
    return row


def expand_cpus(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def proc_starttime(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            s = fh.read()
        return int(s[s.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def alive(pid: int, start: int | None = None) -> bool:
    st = proc_starttime(pid)
    return st is not None and (start is None or st == start)


def block_order(seed: str, b: int, with_s8: bool) -> list[str]:
    """Deterministic shuffle of one block, from the committed campaign seed."""
    runs = list(BASE_BLOCK) + (["S8"] if with_s8 else [])
    h = hashlib.blake2b(f"{seed}\x1forder\x1f{b}".encode(), digest_size=16).digest()
    random.Random(int.from_bytes(h, "big")).shuffle(runs)
    return runs


def make_schedule(seed: str, blocks: int, run_len: int, s8_every: int = 2) -> list[dict]:
    sched = []
    for b in range(blocks):
        with_s8 = s8_every > 0 and (b % s8_every == 1)
        for j, cond in enumerate(block_order(seed, b, with_s8)):
            sched.append({"block": b, "run": j, "cond": cond,
                          "reps": [f"b{b:02d}-r{j}-{cond}-{k}" for k in range(run_len)]})
    return sched


def parse_bench_jsonl(text: str) -> dict:
    """Pull llama-bench -o jsonl result lines out of mixed bench_canonical output."""
    res: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{") and "samples_ts" in line):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = "pp" if d.get("n_gen", 0) == 0 and d.get("n_prompt", 0) > 0 else ("tg" if d.get("n_prompt", 0) == 0 else "mixed")
        res[key] = {"avg_ts": d.get("avg_ts"), "stddev_ts": d.get("stddev_ts"),
                    "samples_ts": d.get("samples_ts"), "samples_ns": d.get("samples_ns"),
                    "n_prompt": d.get("n_prompt"), "n_gen": d.get("n_gen"),
                    "build_commit": d.get("build_commit"), "test_time": d.get("test_time"),
                    "n_threads": d.get("n_threads")}
    return res


# ---------------------------------------------------------------- host probes
def find_gpu_sysfs() -> str | None:
    base = "/sys/class/drm"
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        dev = os.path.join(base, name, "device")
        if not re.fullmatch(r"card\d+", name):
            continue
        try:
            with open(os.path.join(dev, "uevent")) as fh:
                if f"PCI_ID={MI210_PCI_ID}" in fh.read().upper():
                    return dev
        except OSError:
            continue
    return None


def kfd_pids() -> list[int]:
    try:
        return sorted(int(p) for p in os.listdir("/sys/class/kfd/kfd/proc") if p.isdigit())
    except OSError:
        return []


def read_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def cpu_times() -> dict[int, tuple[int, int]]:
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            m = re.match(r"cpu(\d+) (.*)", line)
            if m:
                v = [int(x) for x in m.group(2).split()]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out[int(m.group(1))] = (sum(v[:8]), idle)
    return out


def busy_frac(t0: dict, t1: dict, cpus: list[int]) -> float | None:
    tot = idle = 0
    for c in cpus:
        if c in t0 and c in t1:
            tot += t1[c][0] - t0[c][0]
            idle += t1[c][1] - t0[c][1]
    return None if tot <= 0 else round(1.0 - idle / tot, 4)


def proc_cpu_snapshot() -> dict[int, tuple[int, int, int, str]]:
    """pid -> (utime+stime ticks, ppid, starttime, comm)."""
    snap = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as fh:
                s = fh.read()
            comm = s[s.index("(") + 1:s.rindex(")")]
            f = s[s.rindex(")") + 2:].split()
            snap[int(p)] = (int(f[11]) + int(f[12]), int(f[1]), int(f[19]), comm)
        except (OSError, ValueError, IndexError):
            continue
    return snap


def descendants(snap: dict, roots: set[int]) -> set[int]:
    own = set(r for r in roots if r in snap)
    changed = True
    while changed:
        changed = False
        for pid, (_, ppid, _, _) in snap.items():
            if pid not in own and ppid in own:
                own.add(pid)
                changed = True
    return own


def listener_owned_by(port: int, pid: int) -> bool:
    inodes = set()
    for tbl in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tbl) as fh:
                next(fh)
                for line in fh:
                    f = line.split()
                    if f[3] == "0A" and int(f[1].split(":")[1], 16) == port:
                        inodes.add(f[9])
        except OSError:
            continue
    if not inodes:
        return False
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            m = re.fullmatch(r"socket:\[(\d+)\]", link)
            if m and m.group(1) in inodes:
                return True
    except OSError:
        return False
    return False


def port_listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http_json(port: int, path: str, body: dict | None = None, timeout: float = 10.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def task_affinities(pid: int) -> list[str]:
    out = set()
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/status") as fh:
                    for line in fh:
                        if line.startswith("Cpus_allowed_list:"):
                            out.add(line.split(":", 1)[1].strip())
            except OSError:
                continue
    except OSError:
        pass
    return sorted(out)


def affinity_within(affs: list[str], allowed: str) -> bool:
    """Every task's Cpus_allowed_list is inside `allowed` (OpenMP may bind threads to single CPUs)."""
    ok = set(expand_cpus(allowed))
    return bool(affs) and all(set(expand_cpus(a)) <= ok for a in affs)


# ---------------------------------------------------------------- sampler
class Sampler(threading.Thread):
    def __init__(self, gpu_dev: str | None, host_sets: dict[str, list[int]], period: float = 1.0):
        super().__init__(daemon=True)
        self.gpu_dev, self.host_sets, self.period = gpu_dev, host_sets, period
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self):
        prev = cpu_times()
        while not self._stop.wait(self.period):
            cur = cpu_times()
            s = {"t": round(time.time(), 2), "kfd": kfd_pids()}
            if self.gpu_dev:
                s["busy"] = read_int(os.path.join(self.gpu_dev, "gpu_busy_percent"))
                s["vram"] = read_int(os.path.join(self.gpu_dev, "mem_info_vram_used"))
            for name, cpus in self.host_sets.items():
                s[f"cpu_{name}"] = busy_frac(prev, cur, cpus)
            prev = cur
            self.samples.append(s)

    def stop(self) -> list[dict]:
        self._stop.set()
        self.join(5)
        return self.samples


def summarize_samples(samples: list[dict], own_kfd: set[int]) -> dict:
    def stats(vals):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        return {"n": len(vals), "min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1],
                "mean": round(sum(vals) / len(vals), 4)}
    kfd_union = sorted({p for s in samples for p in s.get("kfd", [])})
    out = {"n_samples": len(samples), "gpu_busy": stats([s.get("busy") for s in samples]),
           "gpu_busy_frac_gt10": (round(sum(1 for s in samples if (s.get("busy") or 0) > 10) / len(samples), 3)
                                  if samples else None),
           "vram": stats([s.get("vram") for s in samples]), "kfd_union": kfd_union,
           "kfd_foreign": [p for p in kfd_union if p not in own_kfd],
           "kfd_own_all_samples": bool(samples) and all(any(p in own_kfd for p in s.get("kfd", [])) for s in samples),
           "kfd_empty_all_samples": bool(samples) and all(not s.get("kfd") for s in samples)}
    for k in [k for k in (samples[0] if samples else {}) if k.startswith("cpu_")]:
        out[k] = stats([s.get(k) for s in samples])
    return out


# ---------------------------------------------------------------- driver
class Driver:
    def __init__(self, a):
        self.a = a
        self.stub = a.mode == "stub"
        self.dry = a.mode == "dry-run"
        self.sess = os.path.join(a.run_dir, "runs", a.session)
        os.makedirs(os.path.join(self.sess, "reps"), exist_ok=True)
        self.timeline_path = os.path.join(self.sess, "timeline.jsonl")
        self.reps_path = os.path.join(self.sess, "reps.jsonl")
        self.children_path = os.path.join(self.sess, "children.pids")
        self.t0 = time.time()
        self.deadline = a.deadline_epoch or (self.t0 + a.max_hours * 3600)
        self.deadline = min(self.deadline, self.t0 + a.max_hours * 3600) - a.teardown_margin_s
        self.server = None          # (Popen, placement, start)
        self.client = None          # (Popen, mode, start, path)
        self.spinner = None         # (Popen, start)
        self.bench = None
        self.gpu_dev = None if self.stub else find_gpu_sysfs()
        self.port = a.port
        self.lockfh = None
        self.stopping = False
        self.rep_seconds: list[float] = []
        self.w2 = a.workload == "w2-q38fn"
        self.cpu_port = a.cpu_port
        self.cpu_server = None      # (Popen of the region-lock wrapper, server pid, server starttime)
        self.cpu_server_info: dict = {}
        self.req_i = 0
        self.prompt_n_ref = None
        self.stop_file = a.stop_file or os.path.join(a.run_dir, "STOP")
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)

    # -- bookkeeping
    def tl(self, step: str, **kw):
        row = {"t": utc(), "elapsed_s": round(time.time() - self.t0, 1), "step": step, **kw}
        with open(self.timeline_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def register(self, pid: int, what: str):
        with open(self.children_path, "a") as fh:
            fh.write(f"{pid} {proc_starttime(pid)} {what}\n")

    def _on_term(self, signum, _frame):
        if self.stopping:
            return
        self.stopping = True
        self.tl("signal", signum=signum)
        raise SystemExit(143)

    # -- process control (own PIDs only)
    def kill_proc(self, p: subprocess.Popen, what: str, group: bool = True, grace: float = 20.0) -> bool:
        if p is None:
            return True
        pid = p.pid
        if p.poll() is None:
            try:
                os.killpg(pid, signal.SIGTERM) if group else p.terminate()
            except ProcessLookupError:
                pass
            t = time.time()
            while p.poll() is None and time.time() - t < grace:
                time.sleep(0.25)
            if p.poll() is None:
                try:
                    os.killpg(pid, signal.SIGKILL) if group else p.kill()
                except ProcessLookupError:
                    pass
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    pass
        dead = not os.path.exists(f"/proc/{pid}") or p.poll() is not None
        self.tl("kill", what=what, pid=pid, verified_dead=dead, rc=p.returncode)
        return dead

    def spawn(self, argv: list[str], what: str, log: str, env: dict | None = None) -> subprocess.Popen:
        fh = open(log, "ab")
        p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=env, start_new_session=True, close_fds=True)
        fh.close()
        self.register(p.pid, what)
        self.tl("spawn", what=what, pid=p.pid, argv=argv[:40])
        return p

    # -- GPU device claim
    def take_device_lock(self) -> bool:
        if self.stub:
            return True
        self.lockfh = open(GPU_DEVICE_LOCK, "a+")
        t = time.time()
        while True:
            try:
                fcntl.flock(self.lockfh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.tl("device_lock", path=GPU_DEVICE_LOCK, held=True)
                return True
            except BlockingIOError:
                if time.time() - t > self.a.lock_wait_s:
                    self.tl("device_lock", path=GPU_DEVICE_LOCK, held=False, waited_s=self.a.lock_wait_s)
                    return False
                time.sleep(5)

    # -- GPU server
    def server_env(self) -> dict:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.path.join(GPU_BUILD, "bin")
        return env

    def start_server(self, placement: str) -> bool:
        if port_listening(self.port):
            self.tl("server_refused", reason=f"port {self.port} already listening")
            return False
        log = os.path.join(self.sess, f"gpu_server.{placement}.{int(time.time())}.log")
        if self.stub:
            argv = [sys.executable, os.path.abspath(__file__), "stub-server", "--port", str(self.port)]
            p = self.spawn(argv, f"stub-server-{placement}", log)
        else:
            p = self.spawn(gpu_server_argv(placement, self.port), f"gpu-server-{placement}", log, self.server_env())
        self.server = (p, placement, time.time())
        t = time.time()
        while time.time() - t < self.a.server_timeout_s:
            if p.poll() is not None:
                self.tl("server_died", rc=p.returncode, log=log)
                return False
            try:
                if http_json(self.port, "/health", timeout=3).get("status") == "ok" and \
                        (self.stub or listener_owned_by(self.port, p.pid)):
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            self.tl("server_health_timeout", log=log)
            return False
        ev = {"placement": placement, "pid": p.pid, "load_s": round(time.time() - t, 1), "log": log}
        if not self.stub:
            try:
                ev["props_build_info"] = http_json(self.port, "/props", timeout=5).get("build_info")
            except Exception as e:
                ev["props_error"] = str(e)[:120]
            try:
                maps = open(f"/proc/{p.pid}/maps").read()
                ev["hip_lib_mapped_from_build"] = os.path.join(GPU_BUILD, "bin", "libggml-hip.so") in maps
            except OSError:
                ev["hip_lib_mapped_from_build"] = False
            time.sleep(2)
            ev["kfd_has_pid"] = p.pid in kfd_pids()
            ev["vram_used"] = read_int(os.path.join(self.gpu_dev, "mem_info_vram_used")) if self.gpu_dev else None
            # ROCr starts one runtime thread with Cpus_allowed 0-191 regardless of the parent's taskset (diag
            # 2026-10-22: 1 of 196 threads). Re-pin every task to the placement, then check.
            ev["affinity_before_pin"] = task_affinities(p.pid)
            ev["pin_rc"] = subprocess.run(["taskset", "-a", "-p", "-c", PLACEMENTS[placement][0], str(p.pid)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            ev["affinity"] = task_affinities(p.pid)
            ok = (ev["kfd_has_pid"] and (ev["vram_used"] or 0) >= MIN_VRAM_BYTES and ev["hip_lib_mapped_from_build"]
                  and ev.get("props_build_info") == GPU_EXPECT_BUILD and affinity_within(ev["affinity"], PLACEMENTS[placement][0]))
            ev["residency_ok"] = ok
            self.tl("server_up", **ev)
            return ok
        self.tl("server_up", **ev)
        return True

    def stop_server(self):
        if self.server:
            self.kill_proc(self.server[0], f"gpu-server-{self.server[1]}")
            self.server = None
            if not self.stub:
                t = time.time()
                while kfd_pids() and time.time() - t < 30:
                    time.sleep(1)

    # -- W2: the persistent Q38FN CPU server (its own 0-95 claim, held for its lifetime)
    def start_cpu_server(self) -> bool:
        if port_listening(self.cpu_port):
            self.tl("cpu_server_refused", reason=f"port {self.cpu_port} already listening")
            return False
        log = os.path.join(self.sess, "q38fn_server.log")
        slot_dir = os.path.join(self.sess, "kv_slots")
        os.makedirs(slot_dir, exist_ok=True)
        if self.stub:
            argv = [sys.executable, os.path.abspath(__file__), "stub-server", "--port", str(self.cpu_port),
                    "--stub-cpu"]
            p = self.spawn(argv, "stub-q38fn-server", log)
        else:
            env = q38fn_env()
            argv = q38fn_launch_argv(self.cpu_port, slot_dir, f"gpu-cpu-noise-w2-{self.a.session}-q38fn", env)
            p = self.spawn(argv, "q38fn-server(region-lock)", log, env)
        self.cpu_server = (p, None, None)    # so every failure path below is reaped by stop_cpu_server
        t = time.time()
        spid = None
        while time.time() - t < self.a.cpu_server_timeout_s:
            if p.poll() is not None:
                self.tl("cpu_server_died", rc=p.returncode, log=log)
                return False
            if self.stub:
                spid = p.pid
            elif spid is None or not alive(spid):
                kids = [k for k in child_pids(p.pid) if proc_comm(k) == "llama-server"]
                spid = kids[0] if kids else None
            if spid:
                self.cpu_server = (p, spid, proc_starttime(spid))
                try:
                    if http_json(self.cpu_port, "/health", timeout=3).get("status") == "ok" and \
                            (self.stub or listener_owned_by(self.cpu_port, spid)):
                        break
                except Exception:
                    pass
            time.sleep(1)
        else:
            self.tl("cpu_server_health_timeout", log=log, server_pid=spid)
            return False
        if not self.stub:
            self.register(spid, "q38fn-llama-server")
        ev = {"wrapper_pid": p.pid, "server_pid": spid, "load_s": round(time.time() - t, 1), "log": log,
              "port": self.cpu_port}
        ok = True
        if not self.stub:
            try:
                ev["props_build_info"] = http_json(self.cpu_port, "/props", timeout=10).get("build_info")
            except Exception as e:
                ev["props_error"] = str(e)[:120]
            try:
                maps = open(f"/proc/{spid}/maps").read()
                ev["cpu_lib_mapped_from_store"] = os.path.join(Q38FN_BINDIR, "libggml-cpu.so") in maps
            except OSError:
                ev["cpu_lib_mapped_from_store"] = False
            ev["affinity"] = task_affinities(spid)
            try:
                ev["environ"] = sorted(x for x in open(f"/proc/{spid}/environ").read().split("\0")
                                       if x.startswith(("GGML_", "OMP_", "KMP_", "LD_LIBRARY_PATH")))
                ev["thp_enabled"] = [l.strip() for l in open(f"/proc/{spid}/status") if l.startswith("THP_enabled")]
            except OSError:
                pass
            plog = os.path.join(self.sess, "q38fn_placement.log")
            pr = subprocess.run(["bash", PLACEMENT_CHECK, str(spid), "--threshold", str(Q38FN_MAX_NODE_SHARE),
                                 "--label", "w2-q38fn"], capture_output=True, text=True, timeout=120)
            with open(plog, "a") as fh:
                fh.write(pr.stdout + pr.stderr)
            ev["placement_rc"], ev["placement_log"] = pr.returncode, plog
            ok = (Q38FN_EXPECT_COMMIT in str(ev.get("props_build_info")) and ev["cpu_lib_mapped_from_store"]
                  and affinity_within(ev["affinity"], Q38FN_CPUS) and pr.returncode == 0)
        else:
            ev["placement_rc"], ev["placement_log"] = 0, None
        ev["identity_ok"] = ok
        self.cpu_server = (p, spid, proc_starttime(spid))
        self.cpu_server_info = ev
        self.tl("cpu_server_up", **ev)
        return ok

    def cpu_server_ok(self) -> bool:
        if not self.cpu_server:
            return False
        p, spid, st = self.cpu_server
        if p.poll() is not None or not spid or not alive(spid, st):
            return False
        return self.stub or listener_owned_by(self.cpu_port, spid)

    def stop_cpu_server(self):
        if not self.cpu_server:
            return
        p, spid, st = self.cpu_server
        self.kill_proc(p, "q38fn-server", grace=self.a.cpu_server_grace_s)
        t = time.time()
        while spid and alive(spid, st) and time.time() - t < 30:
            time.sleep(1)
        if spid and alive(spid, st):
            try:
                os.kill(spid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(2)
        self.tl("cpu_server_down", server_pid=spid, verified_dead=not (spid and alive(spid, st)))
        self.cpu_server = None

    def w2_request(self, body: dict) -> dict:
        t = time.time()
        try:
            r = http_json(self.cpu_port, "/completion", body, timeout=self.a.request_timeout_s)
            tm = r.get("timings") or {}
            return {"ok": True, "t": utc(t), "wall": round(time.time() - t, 3), "prompt_n": tm.get("prompt_n"),
                    "prompt_per_second": tm.get("prompt_per_second"), "predicted_n": tm.get("predicted_n"),
                    "predicted_per_second": tm.get("predicted_per_second"), "draft_n": tm.get("draft_n"),
                    "draft_n_accepted": tm.get("draft_n_accepted")}
        except Exception as e:
            return {"ok": False, "t": utc(t), "wall": round(time.time() - t, 3), "error": str(e)[:160]}

    def w2_work(self) -> dict:
        """One W2 rep: 1 prefill request (unique seeded prompt), then W2_DECODES_PER_REP fixed decode requests."""
        reqs = {"pp": [], "tg": []}
        self.req_i += 1
        reqs["pp"].append(self.w2_request(w2_prefill_body(self.a.w2_seed, self.req_i)))
        for _ in range(W2_DECODES_PER_REP):
            reqs["tg"].append(self.w2_request(w2_decode_body(self.a.w2_seed)))
        res = {}
        for m, key in (("pp", "prompt_per_second"), ("tg", "predicted_per_second")):
            ok = [q for q in reqs[m] if q.get("ok") and q.get(key)]
            if len(ok) == len(reqs[m]):
                res[m] = {"samples_ts": [q[key] for q in ok], "avg_ts": sum(q[key] for q in ok) / len(ok),
                          "prompt_n": [q.get("prompt_n") for q in ok], "predicted_n": [q.get("predicted_n") for q in ok],
                          "draft_n": [q.get("draft_n") for q in ok],
                          "draft_n_accepted": [q.get("draft_n_accepted") for q in ok]}
        rc = 0 if all(q.get("ok") for v in reqs.values() for q in v) else 1
        return {"rc": rc, "res": res, "requests": reqs}

    # -- GPU client load
    def start_client(self, mode: str) -> bool:
        path = os.path.join(self.sess, f"gpu_client.{mode}.{int(time.time())}.jsonl")
        cpus = PLACEMENTS[self.server[1]][0]
        argv = [sys.executable, os.path.abspath(__file__), "client", "--port", str(self.port),
                "--client-mode", mode, "--out", path, "--seed", self.a.seed]
        if not self.stub:
            argv = ["taskset", "-c", cpus] + argv
        p = self.spawn(argv, f"gpu-client-{mode}", path + ".log")
        self.client = (p, mode, time.time(), path)
        # lead-in: the load must be observably on before rep 1 (positive control)
        t = time.time()
        while time.time() - t < self.a.leadin_max_s:
            if p.poll() is not None:
                self.tl("client_died", rc=p.returncode)
                return False
            busy = read_int(os.path.join(self.gpu_dev, "gpu_busy_percent")) if self.gpu_dev else None
            if time.time() - t >= self.a.leadin_min_s and (self.stub or (busy or 0) > 10):
                self.tl("client_on", mode=mode, pid=p.pid, leadin_s=round(time.time() - t, 1), busy=busy)
                return True
            time.sleep(1)
        self.tl("client_leadin_timeout", mode=mode)
        return False

    def stop_client(self):
        if not self.client:
            return
        self.kill_proc(self.client[0], f"gpu-client-{self.client[1]}", grace=5)
        self.client = None
        # settle: no slot processing (the server may finish an in-flight request)
        t = time.time()
        while time.time() - t < self.a.settle_max_s and self.server:
            try:
                slots = http_json(self.port, "/slots", timeout=3)
                if not any(s.get("is_processing") for s in (slots or [])):
                    break
            except Exception:
                pass
            time.sleep(1)
        self.tl("client_settled", waited_s=round(time.time() - t, 1))

    # -- SMT spinner control
    def start_spinners(self):
        cpus = expand_cpus(SPIN_CPUS)
        loop = "while :; do :; done"
        inner = " & ".join(f"taskset -c {c} bash -c '{loop}'" for c in cpus) + " & wait"
        if self.stub:
            inner = " & ".join("sleep 3600" for _ in cpus) + " & wait"
        p = self.spawn(["bash", "-c", inner], "spinners", os.path.join(self.sess, "spinners.log"))
        self.spinner = (p, time.time())
        time.sleep(2)
        self.tl("spinners_on", pid=p.pid, cpus=SPIN_CPUS)

    def stop_spinners(self):
        if self.spinner:
            self.kill_proc(self.spinner[0], "spinners", grace=5)
            self.spinner = None

    # -- condition state machine
    def ensure(self, cond: str) -> bool:
        placement, mode, spin = CONDITIONS[cond]
        self.stop_client()
        if not spin:
            self.stop_spinners()
        if self.server and self.server[1] != placement:
            self.stop_server()
        if placement and not self.server:
            if not self.start_server(placement):
                return False
        if spin and not self.spinner:
            self.start_spinners()
        if mode and not self.start_client(mode):
            return False
        if placement is None and not self.stub:
            ok = not kfd_pids()
            time.sleep(1)
            ok = ok and not kfd_pids()
            if not ok:
                self.tl("kfd_not_empty", cond=cond, kfd=kfd_pids())
        if cond == "G1":
            time.sleep(3)
        self.tl("condition_ready", cond=cond)
        if not self.stub and not self.dry:
            self.rocm_smi_snapshot(cond)
        return True

    def rocm_smi_snapshot(self, cond: str):
        try:
            out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"], capture_output=True, text=True,
                                 timeout=30).stdout
            busy = re.findall(r"GPU use \(%\): (\d+)", out)
            self.tl("rocm_smi", cond=cond, gpu_use_pct=[int(b) for b in busy])
        except Exception as e:
            self.tl("rocm_smi", cond=cond, error=str(e)[:120])

    # -- one CPU rep
    def bench_argv(self, logdir: str) -> list[str]:
        if self.stub:
            fake = {"n_prompt": 512, "n_gen": 0, "avg_ts": 500.0, "stddev_ts": 1.0, "samples_ts": [499.0, 500.0, 501.0],
                    "samples_ns": [1, 1, 1], "build_commit": "stub"}
            fake2 = dict(fake, n_prompt=0, n_gen=128, avg_ts=50.0, samples_ts=[49.9, 50.0, 50.1])
            return ["bash", "-c", f"sleep {self.a.stub_rep_s}; echo '{json.dumps(fake)}'; echo '{json.dumps(fake2)}'"]
        return ["bash", BENCH_CANONICAL, "-m", CPU_MODEL, *BENCH_ARGS, "--log-dir", logdir, "--", "-o", "jsonl"]

    def one_rep(self, rep_id: str, cond: str, block: int, run: int, idx: int, phase: str) -> dict:
        logdir = os.path.join(self.sess, "reps", rep_id)
        os.makedirs(logdir, exist_ok=True)
        log = os.path.join(logdir, "bench.out")
        own_kfd = {self.server[0].pid} if self.server else set()
        sampler = Sampler(self.gpu_dev, {"q3": expand_cpus("184-191"), "n1": expand_cpus("136-143")},
                          self.a.sample_period_s)
        snap0 = proc_cpu_snapshot()
        cpu0 = cpu_times()
        t0 = time.time()
        sampler.start()
        timed_out = False
        w2 = None
        if self.w2:
            w2 = self.w2_work()
            p = None
        else:
            p = self.spawn(self.bench_argv(logdir), f"bench-{rep_id}", log)
            self.bench = p
            try:
                p.wait(self.a.rep_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.kill_proc(p, f"bench-{rep_id}-timeout")
            self.bench = None
        t1 = time.time()
        samples = sampler.stop()
        snap1 = proc_cpu_snapshot()
        cpu1 = cpu_times()
        roots = {os.getpid()} | ({p.pid} if p else set())
        for h in (self.server, self.client, self.spinner, self.cpu_server):
            if h:
                roots.add(h[0].pid)
        own = descendants(snap1, roots) | descendants(snap0, roots)
        hz = os.sysconf("SC_CLK_TCK")
        foreign = {}
        for pid, (ticks, _, st, comm) in snap1.items():
            if pid in own or pid not in snap0 or snap0[pid][2] != st:
                continue
            d = ticks - snap0[pid][0]
            if d > 0:
                foreign[f"{pid}:{comm}"] = d
        wall = max(t1 - t0, 1e-6)
        foreign_cores = round(sum(foreign.values()) / hz / wall, 3)
        top = sorted(foreign.items(), key=lambda kv: -kv[1])[:5]
        if w2 is not None:
            res, rc = w2["res"], w2["rc"]
            with open(log, "w") as fh:
                json.dump(w2["requests"], fh)
            place_path, place_rc = self.cpu_server_info.get("placement_log"), self.cpu_server_info.get("placement_rc")
        else:
            text = open(log, errors="replace").read() if os.path.exists(log) else ""
            res, rc = parse_bench_jsonl(text), p.returncode
            place = re.search(r"Placement proof: (\S+) \(exit (\d+)\)", text)
            place_path, place_rc = (place.group(1), int(place.group(2))) if place else (None, None)
        smry = summarize_samples(samples, own_kfd)
        row = {"schema": SCHEMA_REP, "session": self.a.session, "rep_id": rep_id, "phase": phase,
               "cond": cond, "block": block, "run": run, "idx": idx, "start": utc(t0), "end": utc(t1),
               "wall_s": round(wall, 2), "rc": rc, "timed_out": timed_out,
               "pp": res.get("pp"), "tg": res.get("tg"),
               "placement_proof": place_path,
               "placement_rc": place_rc,
               "gpu": smry, "foreign_cores": foreign_cores,
               "foreign_top": [[k, round(v / hz, 2)] for k, v in top],
               "host_cpu_busy_all": busy_frac(cpu0, cpu1, list(range(192))),
               "server": None if not self.server else {"pid": self.server[0].pid, "placement": self.server[1]},
               "client_mode": None if not self.client else self.client[1],
               "spinners": bool(self.spinner), "stub": self.stub}
        if self.w2:
            row["workload"] = "w2-q38fn"
            row["cpu_server"] = {"pid": self.cpu_server[1] if self.cpu_server else None, "port": self.cpu_port,
                                 "ok_after_rep": self.cpu_server_ok()}
            row["prompt_n_ref"] = self.prompt_n_ref
        row["valid"], row["why_invalid"] = self.validate(row)
        self_hash(row)
        with open(self.reps_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.tl("rep", rep_id=rep_id, cond=cond, phase=phase, rc=rc, wall_s=row["wall_s"],
                tg=(res.get("tg") or {}).get("avg_ts"), pp=(res.get("pp") or {}).get("avg_ts"),
                valid=row["valid"], why=row["why_invalid"], foreign_cores=foreign_cores,
                busy_med=(smry.get("gpu_busy") or {}).get("median"))
        if phase == "measure":
            self.rep_seconds.append(wall)
        return row

    def validate(self, r: dict) -> tuple[bool, list[str]]:
        why = []
        if r["rc"] != 0 or r["timed_out"]:
            why.append(f"bench rc={r['rc']} timed_out={r['timed_out']}")
        if not (r.get("tg") and r.get("pp")):
            why.append("missing pp/tg jsonl rows")
        if r.get("workload") == "w2-q38fn":
            why += w2_controls(r)
        if r["stub"]:
            return (not why, why)
        if r.get("placement_rc") not in (0,):
            why.append(f"placement proof rc={r.get('placement_rc')}")
        g = r["gpu"]
        if (g.get("n_samples") or 0) < 2:
            why.append("fewer than 2 in-window GPU samples")
        placement, mode, spin = CONDITIONS[r["cond"]]
        if placement is None:
            if not g.get("kfd_empty_all_samples"):
                why.append(f"KFD not empty during a GPU-idle rep: {g.get('kfd_union')}")
        else:
            if not g.get("kfd_own_all_samples"):
                why.append("own server pid not in KFD on every sample")
            if ((g.get("vram") or {}).get("min") or 0) < MIN_VRAM_BYTES:
                why.append("VRAM below residency floor")
            if g.get("kfd_foreign"):
                why.append(f"foreign KFD pid(s) {g['kfd_foreign']}")
        if mode and (g.get("gpu_busy_frac_gt10") or 0) < 0.5:
            why.append(f"GPU load did not fire: busy>10% on {g.get('gpu_busy_frac_gt10')} of samples")
        if r["cond"] == "G1" and ((g.get("gpu_busy") or {}).get("median") or 0) > 5:
            why.append("G1 not idle: median busy > 5%")
        if spin and ((g.get("cpu_q3") or {}).get("median") or 0) < 0.9:
            why.append("spinners did not occupy 184-191")
        if r["foreign_cores"] > self.a.foreign_exclude_cores:
            why.append(f"foreign load {r['foreign_cores']} cores > {self.a.foreign_exclude_cores}")
        return (not why, why)

    # -- session
    def preflight(self) -> dict:
        rec = {"schema": SCHEMA_RUN, "session": self.a.session, "mode": self.a.mode, "workload": self.a.workload,
               "start": utc(self.t0),
               "deadline": utc(self.deadline), "code_root": CODE_ROOT, "seed": self.a.seed,
               "blocks_target": self.a.blocks, "blocks_min": self.a.blocks_min, "run_len": self.a.run_len,
               "s8_every": self.a.s8_every, "prereg": self.a.prereg, "prereg_sha256": sha256_file(self.a.prereg)
               if self.a.prereg else None}
        try:
            rec["code_head"] = subprocess.run(["git", "-C", CODE_ROOT, "rev-parse", "HEAD"], capture_output=True,
                                              text=True, timeout=10).stdout.strip()
        except Exception:
            rec["code_head"] = None
        rec["driver_sha256"] = sha256_file(os.path.abspath(__file__))
        rec["uptime_s"] = float(open("/proc/uptime").read().split()[0])
        rec["host_health_tier"] = "fresh(<1wk)" if rec["uptime_s"] < 7 * 86400 else \
            "STALE(>=1wk): relative estimands valid; session floor not quotable as the host quiet floor"
        rec["loadavg"] = open("/proc/loadavg").read().strip()
        rec["kernel"] = os.uname().release
        for k, p in {"thp_enabled": "/sys/kernel/mm/transparent_hugepage/enabled",
                     "thp_defrag": "/sys/kernel/mm/transparent_hugepage/defrag"}.items():
            try:
                rec[k] = open(p).read().strip()
            except OSError:
                rec[k] = None
        problems = []
        if not self.stub:
            rec["gpu_sysfs"] = self.gpu_dev
            if not self.gpu_dev:
                problems.append(f"MI210 ({MI210_PCI_ID}) sysfs not found")
            cpu_files = ((*Q38FN_SHARDS, Q38FN_MTP, os.path.join(Q38FN_BINDIR, "llama-server"), REGION_LOCK,
                          NUMA_EVICT, PLACEMENT_CHECK, RAM_ADMIT) if self.w2 else (CPU_MODEL, BENCH_CANONICAL))
            for f in (*cpu_files, GPU_MODEL, GPU_DRAFT, GPU_TEMPLATE, VERIFY_LINKAGE,
                      os.path.join(GPU_BUILD, "bin", "llama-server")):
                if not os.path.exists(f):
                    problems.append(f"missing {f}")
            rec["gpu_server_sha256"] = sha256_file(os.path.join(GPU_BUILD, "bin", "llama-server"))
            rec["gpu_hip_sha256"] = sha256_file(os.path.join(GPU_BUILD, "bin", "libggml-hip.so.0.16.0"))
            lk = subprocess.run(["bash", VERIFY_LINKAGE, os.path.join(GPU_BUILD, "bin", "llama-server"), GPU_BUILD],
                                capture_output=True, text=True, timeout=120)
            rec["gpu_linkage_rc"] = lk.returncode
            rec["gpu_linkage_tail"] = lk.stdout[-400:]
            if lk.returncode != 0:
                problems.append("verify_ggml_linkage FAILED for the GPU build")
            if self.w2:
                problems += self.preflight_w2(rec)
            else:
                bc = subprocess.run(["bash", BENCH_CANONICAL, "-m", CPU_MODEL, *BENCH_ARGS, "--log-dir",
                                     os.path.join(self.sess, "bench-dry"), "--dry-run", "--", "-o", "jsonl"],
                                    capture_output=True, text=True, timeout=300)
                rec["bench_canonical_dry_rc"] = bc.returncode
                m = re.search(r"Cmd:\s+(.*)", bc.stdout + bc.stderr)
                rec["bench_cmd"] = m.group(1) if m else None
                b = re.search(r"Binary:\s+(\S+)", bc.stdout + bc.stderr)
                rec["bench_binary"] = b.group(1) if b else None
                rec["bench_binary_sha256"] = sha256_file(b.group(1)) if b else None
                if bc.returncode != 0 or not m or "-o jsonl" not in m.group(1) or "-t 96" not in m.group(1):
                    problems.append(f"bench_canonical dry-run rc={bc.returncode} cmd={rec['bench_cmd']}")
            rec["kfd_at_start"] = kfd_pids()
            rec["port_free"] = not port_listening(self.port)
            if not rec["port_free"]:
                problems.append(f"port {self.port} is listening")
            if rec["kfd_at_start"] and self.a.mode == "run":
                problems.append(f"KFD not empty at start: {rec['kfd_at_start']}")
            try:
                rec["region_lock_status"] = subprocess.run(
                    ["/workspace/repos/epyc-orchestrator/scripts/region-lock", "status"], capture_output=True,
                    text=True, timeout=30).stdout
            except Exception as e:
                rec["region_lock_status"] = f"error {e}"
        if self.w2:
            rec["w2"] = {"seed": self.a.w2_seed, "cpu_port": self.cpu_port, "with_g3": self.a.w2_with_g3,
                         "decode_body_sha256": hashlib.sha256(json.dumps(w2_decode_body(self.a.w2_seed),
                                                                         sort_keys=True).encode()).hexdigest(),
                         "prefill_words": W2_PREFILL_WORDS, "decode_n": W2_DECODE_N,
                         "decodes_per_rep": W2_DECODES_PER_REP,
                         "server_argv": q38fn_launch_argv(self.cpu_port, os.path.join(self.sess, "kv_slots"),
                                                          f"gpu-cpu-noise-w2-{self.a.session}-q38fn", q38fn_env())}
        rec["problems"] = problems
        return rec

    def preflight_w2(self, rec: dict) -> list[str]:
        problems = []
        store = os.path.realpath(Q38FN_STORE_LINK)
        rec["q38fn_store_resolved"] = store
        if store != Q38FN_BINDIR:
            problems.append(f"{Q38FN_STORE_LINK} resolves to {store}, not the production v10 store {Q38FN_BINDIR}")
        srv = os.path.join(Q38FN_BINDIR, "llama-server")
        rec["q38fn_server_sha256"] = sha256_file(srv)
        rec["q38fn_ggml_cpu_sha256"] = sha256_file(os.path.join(Q38FN_BINDIR, "libggml-cpu.so"))
        rec["q38fn_mtp_bytes"] = os.path.getsize(Q38FN_MTP) if os.path.exists(Q38FN_MTP) else None
        rec["q38fn_trunk_bytes"] = sum(os.path.getsize(f) for f in Q38FN_SHARDS if os.path.exists(f))
        lk = subprocess.run(["bash", VERIFY_LINKAGE, srv, Q38FN_BINDIR], capture_output=True, text=True,
                            timeout=120, env=q38fn_env())
        rec["q38fn_linkage_rc"], rec["q38fn_linkage_tail"] = lk.returncode, lk.stdout[-400:]
        if lk.returncode != 0:
            problems.append("verify_ggml_linkage FAILED for the Q38FN store binary")
        ra = subprocess.run(["bash", RAM_ADMIT, str(Q38FN_RAM_NEED_GIB)], capture_output=True, text=True, timeout=60)
        rec["ram_admit"] = {"rc": ra.returncode, "out": ra.stdout.strip()[:200], "need_gib": Q38FN_RAM_NEED_GIB}
        if ra.returncode != 0:
            problems.append(f"ram_admit refused: {ra.stdout.strip()[:160]}")
        rec["prod_q38fn_listening"] = port_listening(PROD_Q38FN_PORT)
        if rec["prod_q38fn_listening"]:
            problems.append(f"production architect_general :{PROD_Q38FN_PORT} is listening (a second Q38FN would "
                            "contend for 0-95 and add 90+ GiB)")
        if port_listening(self.cpu_port):
            problems.append(f"Q38FN port {self.cpu_port} is listening")
        return problems

    def run(self) -> int:
        rec = self.preflight()
        sched = (make_schedule_w2(self.a.blocks, self.a.run_len, self.a.w2_with_g3) if self.w2
                 else make_schedule(self.a.seed, self.a.blocks, self.a.run_len, self.a.s8_every))
        sched_blob = json.dumps(sched, sort_keys=True)
        rec["schedule_sha256"] = hashlib.sha256(sched_blob.encode()).hexdigest()
        with open(os.path.join(self.sess, "schedule.json"), "w") as fh:
            fh.write(sched_blob)
        est = self.a.est_rep_s * sum(len(r["reps"]) for r in sched) + self.a.est_switch_s * len(sched) + \
            (420 if self.w2 else 600)
        rec["estimated_s"] = est
        with open(os.path.join(self.sess, "RUN_RECORD.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        self.tl("preflight", problems=rec["problems"], est_h=round(est / 3600, 2),
                n_runs=len(sched), n_reps=sum(len(r["reps"]) for r in sched), host_health=rec["host_health_tier"])
        if self.dry:
            for r in sched:
                self.tl("plan", block=r["block"], run=r["run"], cond=r["cond"], argv=self.plan_argv(r["cond"]))
            return 0 if not rec["problems"] else 2
        if rec["problems"]:
            return 2
        if not self.take_device_lock():
            return 4
        # ---- W2: the persistent CPU server (own 0-95 claim), one relaunch on a failed identity/placement proof
        if self.w2:
            if not self.start_cpu_server():
                self.stop_cpu_server()
                self.tl("cpu_server_retry")
                if not self.start_cpu_server():
                    self.tl("abort", reason="Q38FN server failed its identity/placement proof twice")
                    return 8
        # ---- warm-up (discarded)
        self.ensure("G0")
        for k in range(self.a.warm_cpu):
            row = self.one_rep(f"warm-cpu-{k}", "G0", -1, -1, k, "warmup")
            if self.w2 and k == 0:
                pn = [n for n in ((row.get("pp") or {}).get("prompt_n") or []) if n]
                self.prompt_n_ref = sorted(pn)[len(pn) // 2] if pn else None
                self.tl("prompt_n_ref", prompt_n_ref=self.prompt_n_ref)
        if not self.ensure("G2"):
            self.tl("abort", reason="GPU warm-up failed (server/residency/client)")
            return 5
        time.sleep(self.a.warm_gpu_s)
        self.one_rep("warm-both-0", "G2", -1, -1, 0, "warmup")
        if not self.w2 or self.a.w2_with_g3:
            if not self.ensure("G3"):
                return 5
            self.one_rep("warm-both-1", "G3", -1, -1, 0, "warmup")
        # ---- blocks
        blocks_plan = self.a.blocks
        done_blocks = 0
        block_start = time.time()
        for r in sched:
            b = r["block"]
            if b >= blocks_plan:
                break
            if r["run"] == 0:
                if b == 1:  # adapt ONCE, from timing only (pre-registered)
                    per_block = time.time() - block_start
                    fits = int((self.deadline - time.time()) // per_block) + 1
                    new = max(self.a.blocks_min, min(blocks_plan, fits))
                    self.tl("adapt", block1_s=round(per_block), fits=fits, blocks_plan=new,
                            note="timing-only, decided before any analysis")
                    blocks_plan = new
                    if b >= blocks_plan:
                        break
                block_start = time.time()
            if os.path.exists(self.stop_file):  # FILLERS rule 2: honoured between items
                self.tl("stop_file", block=b, run=r["run"])
                break
            need = (sum(self.rep_seconds) / len(self.rep_seconds) if self.rep_seconds else self.a.est_rep_s)
            if time.time() + need * len(r["reps"]) + self.a.est_switch_s > self.deadline:
                self.tl("deadline_stop", block=b, run=r["run"])
                break
            if not self.ensure(r["cond"]):
                self.tl("condition_failed", cond=r["cond"], block=b)
                if not self.ensure(r["cond"]):
                    self.tl("abort", reason=f"condition {r['cond']} failed twice")
                    return 6
            for k, rid in enumerate(r["reps"]):
                row = self.one_rep(rid, r["cond"], b, r["run"], k, "measure")
                if not row["valid"] and (row["rc"] != 0 or row["timed_out"]):
                    row = self.one_rep(rid + "-retry", r["cond"], b, r["run"], k, "measure")
                if self.w2 and not self.cpu_server_ok():
                    self.tl("abort", reason="Q38FN server died or lost its port", block=b, run=r["run"])
                    self.blocks_done = done_blocks
                    return 8
            if r["run"] == max(x["run"] for x in sched if x["block"] == b):
                done_blocks = b + 1
                self.tl("block_done", block=b)
        self.blocks_done = done_blocks
        return 0 if done_blocks >= self.a.blocks_min else 7

    def plan_argv(self, cond: str) -> dict:
        placement, mode, spin = CONDITIONS[cond]
        out = {"gpu_server": gpu_server_argv(placement, self.port) if placement else None,
               "client": mode, "spinners": SPIN_CPUS if spin else None}
        if self.w2:
            out["cpu_work"] = {"prefill": 1, "decode": W2_DECODES_PER_REP, "port": self.cpu_port}
        else:
            out["bench"] = self.bench_argv(os.path.join(self.sess, "reps", "<rep>"))
        return out

    def cleanup(self):
        for fn in (self.stop_client, self.stop_spinners, self.stop_server, self.stop_cpu_server):
            try:
                fn()
            except Exception as e:
                self.tl("cleanup_error", fn=fn.__name__, error=str(e)[:200])
        if self.bench and self.bench.poll() is None:
            self.kill_proc(self.bench, "bench-at-cleanup")
        # belt and braces: anything we registered that is still alive (start time must match)
        if os.path.exists(self.children_path):
            for line in open(self.children_path):
                f = line.split()
                if len(f) >= 2 and f[1] != "None" and alive(int(f[0]), int(f[1])):
                    try:
                        os.killpg(int(f[0]), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    time.sleep(1)
                    self.tl("registry_kill", pid=int(f[0]), what=" ".join(f[2:]), dead=not alive(int(f[0]), int(f[1])))
        if self.lockfh:
            fcntl.flock(self.lockfh, fcntl.LOCK_UN)
            self.lockfh.close()
            self.tl("device_lock", held=False)


# ---------------------------------------------------------------- client (GPU load generator)
WORDS = ("alpha beta gamma delta epsilon river mountain signal vector kernel matrix lattice ocean forest "
         "copper silver engine harbor window market theory garden circuit planet orbit carbon marble "
         "lantern meadow quartz falcon ember willow canyon prism tundra nebula cobalt saffron").split()


def prefill_prompt(seed: str, i: int, n_words: int) -> str:
    rng = random.Random(f"{seed}:{i}")
    return f"[{i}] " + " ".join(rng.choice(WORDS) for _ in range(n_words)) + "\nSummarize the text above in one line."


def client_main(a) -> int:
    i = 0
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with open(a.out, "a") as fh:
        while not stop["v"]:
            if a.client_mode == "decode":
                body = {"prompt": "Write a long, detailed technical essay about how compilers optimize loops.",
                        "n_predict": 512, "temperature": 0, "top_k": 1, "seed": 7, "ignore_eos": True,
                        "cache_prompt": False}
            else:
                body = {"prompt": prefill_prompt(a.seed, i, a.prefill_words), "n_predict": 8, "temperature": 0,
                        "top_k": 1, "seed": 7, "cache_prompt": False}
            t = time.time()
            try:
                r = http_json(a.port, "/completion", body, timeout=900)
                tm = r.get("timings") or {}
                row = {"i": i, "t": utc(t), "wall": round(time.time() - t, 3), "prompt_n": tm.get("prompt_n"),
                       "prompt_per_second": tm.get("prompt_per_second"), "predicted_n": tm.get("predicted_n"),
                       "predicted_per_second": tm.get("predicted_per_second"), "draft_n": tm.get("draft_n"),
                       "draft_n_accepted": tm.get("draft_n_accepted")}
            except Exception as e:
                row = {"i": i, "t": utc(t), "error": str(e)[:160]}
                time.sleep(1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            i += 1
    return 0


class _Stub(http.server.BaseHTTPRequestHandler):
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok"} if self.path == "/health" else [{"id": 0, "is_processing": False}])

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if getattr(self.server, "stub_cpu", False):
            req = json.loads(body or b"{}")
            n_words = len(str(req.get("prompt", "")).split())
            time.sleep(0.05)
            self._send({"content": "x", "timings": {
                "prompt_n": int(n_words * 1.33), "prompt_per_second": 230.0 * random.uniform(0.995, 1.005),
                "predicted_n": int(req.get("n_predict", 1)), "predicted_per_second": 40.0 * random.uniform(0.995, 1.005),
                "draft_n": 100, "draft_n_accepted": 82}})
            return
        time.sleep(0.2)
        self._send({"content": "x", "timings": {"prompt_n": 10, "predicted_n": 8, "predicted_per_second": 1.0}})

    def log_message(self, *_):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "dry-run", "stub", "client", "stub-server"])
    ap.add_argument("--run-dir", default="/mnt/raid0/llm/tmp/gpu-cpu-noise-20261021")
    ap.add_argument("--session", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--prereg", default=None, help="path of the pre-registration doc (sha256 recorded)")
    ap.add_argument("--seed", default="gpu-cpu-noise-20261021")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--blocks-min", type=int, default=7)
    ap.add_argument("--run-len", type=int, default=5)
    ap.add_argument("--s8-every", type=int, default=2)
    ap.add_argument("--deadline-epoch", type=float, default=None)
    ap.add_argument("--max-hours", type=float, default=3.5)
    ap.add_argument("--teardown-margin-s", type=float, default=180)
    ap.add_argument("--port", type=int, default=GPU_PORT)
    ap.add_argument("--warm-cpu", type=int, default=2)
    ap.add_argument("--warm-gpu-s", type=float, default=20)
    ap.add_argument("--est-rep-s", type=float, default=30)
    ap.add_argument("--est-switch-s", type=float, default=20)
    ap.add_argument("--rep-timeout-s", type=float, default=420)
    ap.add_argument("--server-timeout-s", type=float, default=300)
    ap.add_argument("--leadin-min-s", type=float, default=8)
    ap.add_argument("--leadin-max-s", type=float, default=90)
    ap.add_argument("--settle-max-s", type=float, default=60)
    ap.add_argument("--sample-period-s", type=float, default=1.0)
    ap.add_argument("--foreign-exclude-cores", type=float, default=4.0)
    ap.add_argument("--lock-wait-s", type=float, default=300)
    ap.add_argument("--stub-rep-s", type=float, default=0.3)
    ap.add_argument("--stop-file", default=None, help="default <run-dir>/STOP")
    # W2-Q38FN (addendum PREREG.ADDENDUM-W2-Q38FN.md)
    ap.add_argument("--workload", choices=WORKLOADS, default="w1")
    ap.add_argument("--cpu-port", type=int, default=Q38FN_PORT)
    ap.add_argument("--w2-seed", default=W2_SEED)
    ap.add_argument("--w2-with-g3", action="store_true", help="ABBA + G3 (NOT in the registered W2 session)")
    ap.add_argument("--cpu-server-timeout-s", type=float, default=600)
    ap.add_argument("--cpu-server-grace-s", type=float, default=45)
    ap.add_argument("--request-timeout-s", type=float, default=300)
    ap.add_argument("--stub-cpu", action="store_true", help=argparse.SUPPRESS)
    # client / stub-server
    ap.add_argument("--client-mode", choices=["decode", "prefill"])
    ap.add_argument("--out")
    ap.add_argument("--prefill-words", type=int, default=12000)
    a = ap.parse_args()
    if a.mode == "client":
        return client_main(a)
    if a.mode == "stub-server":
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), _Stub)
        srv.stub_cpu = a.stub_cpu
        srv.serve_forever()
        return 0
    d = Driver(a)
    rc, reason = 1, "exception"
    d.blocks_done = 0
    try:
        rc = d.run()
        reason = {0: "complete", 2: "preflight-problems", 4: "refused:device-lock", 5: "gpu-warmup-failed",
                  6: "condition-failed", 7: "partial(<blocks_min)", 8: "cpu-server-failed"}.get(rc, f"rc{rc}")
    except SystemExit as e:
        rc, reason = int(e.code or 143), "signalled"
    except Exception as e:  # never leave children behind
        d.tl("exception", error=repr(e)[:400])
    finally:
        d.cleanup()
        end = {"rc": rc, "reason": reason, "time": utc(), "blocks_done": getattr(d, "blocks_done", 0),
               "session": a.session, "mode": a.mode, "workload": a.workload}
        with open(os.path.join(d.sess, "END.json"), "w") as fh:
            json.dump(end, fh)
        d.tl("end", **end)
    return rc


if __name__ == "__main__":
    sys.exit(main())
