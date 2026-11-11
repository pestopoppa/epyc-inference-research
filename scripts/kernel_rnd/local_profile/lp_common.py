#!/usr/bin/python3
"""lp_common: shared library of the C6-LOCAL-SERVING-PROFILE / C6-LOCAL-CODE28K rows (CPU llama-server, Q38FN + MTP).

Reconstructed 2026-11-11 from its callers (c7-local-{serving-profile,code28k}-20261106/job.py) after the original was
reclaimed from tmp. Row dirs carry a SHA-sealed copy; this file is the source of truth.

Contract used by the jobs:
  constants   STORE, STORE_SOURCE, MODEL, MTP, N_PREDICT, CODE_SRC, FENCE
  helpers     now, log, should_stop, granted_physical, seeded_orders, tci, valid, precond
  Ledger      jsonl ledger: .rows, .done(unit_id), .add(row)
  Server      one fenced llama-server per visit: start/complete/props/placement/stop
  build_code_prompt(server, out_path, meta_path)  complete-file code prompt, <= TARGET_TOKENS, no EOS at the boundary

Process rules: children are ordinary Popen children in the caller's process group (no start_new_session, no killpg);
stop is terminate -> wait -> kill on the captured pid only.
"""
import hashlib
import json
import math
import os
import random
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

STORE = "/mnt/raid0/llm/kernels/builds/cpu-20261109-7f1bd044b-c10fold"  # C10 champion CPU store
STORE_SOURCE = "C10 7f1bd044b (llama.cpp-experimental-c10-fold-20261109), store cpu-20261109-7f1bd044b-c10fold"
MODEL = ("/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
MTP = "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
FENCE = "/mnt/raid0/llm/tmp/compute-flags/bin/job_templates/job_fence.py"
N_PREDICT = 256
TARGET_TOKENS = 28000
MAX_BUILD_TRIES = 6
CODE_SRC = ""  # set by the row job (inputs/code_src.jsonl: {"path","text"} per line, sorted)
ENV = {"GGML_FA_SPLIT_KV": "0", "GGML_FUSED_DECODE_OFF": "1", "GGML_IQK": "1", "GGML_NOHUGEPAGE_PROCESS": "1",
       "OMP_DYNAMIC": "false", "OMP_PLACES": "cores", "OMP_PROC_BIND": "spread", "OMP_WAIT_POLICY": "active"}

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def should_stop():
    stopf = os.environ.get("FILLER_STOPFILE")
    dl = os.environ.get("FILLER_DEADLINE")
    return bool((stopf and os.path.exists(stopf)) or (dl and time.time() > int(dl)))


def expand_cpus(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def compress_cpus(cpus):
    cpus, out, i = sorted(set(cpus)), [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def fold_physical(cpus, n_phys=96):
    """SMT-fold: cpu c and c+96 are one physical core; return sorted physical ids."""
    return sorted({c % n_phys for c in cpus})


def granted_physical(environ=None, affinity=None):
    environ = os.environ if environ is None else environ
    if environ.get("FILLER_CPUS"):
        cpus = expand_cpus(environ["FILLER_CPUS"])
    else:
        cpus = sorted(affinity if affinity is not None else os.sched_getaffinity(0))
    return fold_physical(cpus)


def seeded_orders(seed, n):
    """n pair orders, exactly half 'AB' and half 'BA', shuffled by random.Random(seed). n must be even."""
    if n % 2:
        raise ValueError("n must be even")
    o = ["AB"] * (n // 2) + ["BA"] * (n // 2)
    random.Random(seed).shuffle(o)
    return o


def tci(ds):
    """Two-sided 95% t-interval of the mean: {n, mean, sd, lo, hi}. n<2 gives mean only."""
    n = len(ds)
    if n == 0:
        return {"n": 0}
    m = sum(ds) / n
    if n < 2:
        return {"n": n, "mean": m}
    sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (n - 1))
    df = n - 1
    t = _T95.get(df) or (2.021 if df <= 40 else 2.0 if df <= 60 else 1.98 if df <= 120 else 1.96)
    h = t * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h}


def valid(r, ref_prompt_n=None):
    """A unit is valid iff it decoded exactly N_PREDICT tokens and (when a reference is known) prompt_n matches."""
    if r.get("predicted_n") != N_PREDICT:
        return False
    return ref_prompt_n is None or r.get("prompt_n") == ref_prompt_n


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def precond(files):
    """rc 0 iff every input file, the store's llama-server and both model files exist and are non-empty; else 5."""
    need = [str(f) for f in files if f] + [STORE + "/bin/llama-server", MODEL, MTP]
    bad = [p for p in need if not os.path.exists(p) or (os.path.isfile(p) and os.path.getsize(p) == 0)]
    for p in bad:
        log(f"precondition missing/empty: {p}")
    return 5 if bad else 0


class Ledger:
    """Append-only jsonl ledger keyed by unit_id; resumable."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        self.rows.append(json.loads(line))
                    except ValueError:
                        pass  # torn trailing line from a hard end

    def done(self, unit_id):
        return any(r.get("unit_id") == unit_id for r in self.rows)

    def add(self, row):
        row = {**row, "ts": now()}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)


def server_argv(store, port, threads, mtp, cpus, model=None, mtp_model=None, fence_dir=None):
    """The launch_local_inference.sh CPU serving argv (np4/262144/f16/no-kv-unified), fenced, pinned to `cpus`."""
    ex = ["/usr/bin/python3", "-I", FENCE, "fence", str(fence_dir), "--"] if fence_dir else []
    a = ex + ["/usr/bin/taskset", "-c", compress_cpus(cpus), "/usr/bin/numactl", "--interleave=all",
              f"{store}/bin/llama-server", "-m", model or MODEL, "-np", "4", "-c", "262144",
              "-t", str(threads), "-tb", str(threads), "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16",
              "--device", "none", "-ngl", "0", "-fa", "on", "--host", "127.0.0.1", "--port", str(port),
              "--metrics", "--slots", "--load-threads", str(len(cpus)), "--no-mmap", "--no-webui"]
    if mtp:
        a += ["-md", mtp_model or MTP, "-ngld", "0", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
              "--device-draft", "none", "--draft-p-min", "0.5"]
    return a + ["--no-kv-unified"]


class Server:
    def __init__(self, store, port, threads, mtp, cpus, outdir, uid):
        self.store, self.port, self.threads, self.mtp = store, port, threads, bool(mtp)
        self.cpus, self.outdir, self.uid = list(cpus), Path(outdir), uid
        self.p = None
        self.base = f"http://127.0.0.1:{port}"

    def start(self, timeout=1800):
        self.outdir.mkdir(parents=True, exist_ok=True)
        argv = server_argv(self.store, self.port, self.threads, self.mtp, self.cpus, fence_dir=self.outdir)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/node"),
               "LD_LIBRARY_PATH": f"{self.store}/bin", "MAX_FILE_MIB": "1024", **ENV}
        if os.environ.get("FILLER_ID"):  # keeper attributes the row's processes by FILLER_ID
            env["FILLER_ID"] = os.environ["FILLER_ID"]
        if os.environ.get("LD_PRELOAD"):
            env["LD_PRELOAD"] = os.environ["LD_PRELOAD"]
        (self.outdir / f"{self.uid}.argv.json").write_text(json.dumps({"argv": argv, "env": env}, indent=1))
        logf = open(self.outdir / f"{self.uid}.server.log", "wb")
        self.p = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT, env=env)  # same process group
        pf = os.environ.get("FILLER_PIDFILE")
        if pf:
            with open(pf, "a") as f:
                f.write(f"{self.p.pid}\n")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if should_stop():
                raise InterruptedError("stop during load")
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited rc={self.p.returncode} during load ({self.uid})")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if b'"ok"' in r.read():
                        return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError(f"server not healthy within {timeout}s ({self.uid})")

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def _post(self, path, body, timeout=7200):
        """POST in a worker thread so STOP/deadline can interrupt a long prefill."""
        res = {}

        def go():
            try:
                req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    res["v"] = json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                res["e"] = e
        th = threading.Thread(target=go, daemon=True)
        th.start()
        while th.is_alive():
            th.join(2)
            if should_stop():
                raise InterruptedError("stop during request")
        if "e" in res:
            raise res["e"]
        return res["v"]

    def complete(self, text, n_predict):
        """Greedy, cache_prompt=false completion; returns timings summary plus content and content_sha256."""
        t0 = time.time()
        d = self._post("/completion", {"prompt": text, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                                       "cache_prompt": False, "stream": False})
        tm = d.get("timings") or {}
        c = d.get("content", "")
        return {"tok_s": tm.get("predicted_per_second"), "prompt_n": tm.get("prompt_n", d.get("tokens_evaluated")),
                "predicted_n": tm.get("predicted_n", d.get("tokens_predicted")),
                "stop_type": d.get("stop_type"), "draft_n": tm.get("draft_n"),
                "draft_n_accepted": tm.get("draft_n_accepted"), "wall_s": round(time.time() - t0, 2),
                "content": c, "content_sha256": hashlib.sha256(c.encode()).hexdigest()}

    def tokenize(self, text, special=True):
        return self._post("/tokenize", {"content": text, "add_special": False, "parse_special": special},
                          timeout=600)["tokens"]

    def detokenize(self, tokens):
        return self._post("/detokenize", {"tokens": tokens}, timeout=600)["content"]

    def props(self):
        p = self._get("/props")
        return {"build_info": p.get("build_info"), "total_slots": p.get("total_slots"),
                "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx")}

    def placement(self):
        """Proof of the running binary: libraries mapped by the server pid must all come from this store."""
        pid = self.p.pid
        try:
            libs = {l.split()[-1] for l in open(f"/proc/{pid}/maps") if "libggml" in l or "libllama" in l}
            aff = sorted(os.sched_getaffinity(pid))
        except OSError:
            return {"pid": pid, "ok": None}
        st = os.path.realpath(self.store) + "/"
        own = sorted(x for x in libs if os.path.realpath(x).startswith(st))
        foreign = sorted(x for x in libs if not os.path.realpath(x).startswith(st))
        return {"pid": pid, "ok": bool(own) and not foreign, "own": own, "foreign": foreign,
                "affinity": compress_cpus(aff)}

    def stop(self):
        p = self.p
        if p is None or p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def is_eos_text(s):
    return any(t in s for t in ("<|endoftext|>", "<|im_end|>", "</s>", "<eos>"))


def build_code_prompt(server, out_path, meta_path, src_path=None, target=TARGET_TOKENS):
    """Whole source files in path order up to <= target tokens, then the next file's header as a continuation cue.
    Each try must: contain no EOS piece, end on a non-EOS token, and a 1-token greedy probe must stop on 'limit'.
    On failure back off one file; after MAX_BUILD_TRIES raise RuntimeError('no non-EOS ...')."""
    rows = [json.loads(l) for l in Path(src_path or CODE_SRC).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["path"])
    chunks = [f"# File: {r['path']}\n{r['text'].rstrip()}\n\n" for r in rows]
    cum, toks = [], 0
    for c in chunks:
        toks += len(server.tokenize(c))
        cum.append(toks)
    k = max(i for i, t in enumerate(cum) if t <= target) + 1 if cum and cum[0] <= target else 0
    if k == 0:
        raise RuntimeError("no non-EOS complete-file boundary: first file exceeds target")
    tries = []
    for _ in range(MAX_BUILD_TRIES):
        if k <= 0:
            break
        nxt = f"# File: {rows[k]['path']}\n" if k < len(rows) else "# File: src/_next.py\n"
        text = "".join(chunks[:k]) + nxt
        ids = server.tokenize(text)
        eos_ids = server.tokenize("<|endoftext|><|im_end|>")
        bad = is_eos_text(text) or not ids or ids[-1] in eos_ids
        stop_type = None
        if not bad:
            r = server.complete(text, 1)
            stop_type = r.get("stop_type")
            bad = stop_type != "limit"
        tries.append({"files": k, "tokens": len(ids), "bad": bad, "stop_type": stop_type})
        if not bad:
            Path(out_path).write_text(text)
            meta = {"prompt_tokens": len(ids), "files": k, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "last_file": rows[k - 1]["path"], "tries": tries, "built_utc": now()}
            Path(meta_path).write_text(json.dumps(meta, indent=1))
            return meta
        k -= 1
    raise RuntimeError(f"no non-EOS complete-file boundary in {MAX_BUILD_TRIES} tries: {tries}")
