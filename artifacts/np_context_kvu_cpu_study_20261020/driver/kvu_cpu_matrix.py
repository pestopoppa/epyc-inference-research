#!/usr/bin/env python3
"""CPU np x context x KV-mode matrix driver (KVU-12 reopen, operator request 2026-10-20).

Successor to ../../np_context_kvu_study_20260924/driver/ (GPU, bash + v7_quality_gate_runner).  That runner
cannot do fixed-length generation, multi-wave, streaming TTFT or shared-prefix prompts, which is exactly what
KVU-8 / this campaign need, so this is a stdlib-only Python driver that keeps the GPU driver's skeleton
(launch -> wait healthy -> load -> stop, per-cell dir, summary.tsv) and the M-4 request shape (/completion,
ignore_eos, n_predict, greedy).  See ../PLAN.md for the spec.

Every launch is the PRODUCTION frontdoor argv (`--verify-against-launcher` re-derives it from
orchestrator_stack.build_server_command and diffs), with only these knobs varied: -np, -c, --kv-unified /
--no-kv-unified, MTP on/off, plus the bench-only substitutions (--port, --slot-save-path dropped so the
production slot dir is never touched, -t/cpu list under a quarter claim).

Process rules (CLAUDE.md): the server PID is captured from Popen and killed only by PID (SIGTERM -> SIGKILL ->
verify); there is no name-pattern kill anywhere.  Run the whole driver under `region-lock run` so it holds the
CPU claim; the server is a child of this driver, which is a child of region-lock.

Write-side evidence: every cell dir has cell.log whose first lines echo the effective argv/env/binary-linkage
and whose last line is `CELL_SUMMARY {json}`; summary.tsv gets one row per cell.
"""
import argparse, atexit, hashlib, json, os, re, signal, statistics, subprocess, sys, threading, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent
KSTORE = Path("/mnt/raid0/llm/kernels/production/cpu")           # symlink -> kernels/builds/cpu-20260921-ffc1bac82/bin
BIN = KSTORE / "llama-server"
MODEL = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
TEMPLATE = "/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja"
LINKAGE = "/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
ORCH = "/mnt/raid0/llm/epyc-orchestrator"
# Canonical production env stack (stack_env._CANONICAL_OMP_ENV + frontdoor LD path); --verify-against-launcher checks it.
PROD_ENV = {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "active", "OMP_DYNAMIC": "false",
            "KMP_BLOCKTIME": "10", "GGML_IQK": "1", "LD_LIBRARY_PATH": "/usr/lib/llvm-20/lib"}
KV_BYTES_PER_TOKEN = 10.9e3     # q8_0, 10 full-attention layers (registry kv_kib_per_token_f16 20.0 x 8.5/16)
HOST_BW_GBS = 446.8             # measured read-sum at t96 (cpu-decode-roofline-program.md)
W_TOTAL_GB, W_ACTIVE1_GB = 37.8, 3.3   # Q8_0 file; ~3B active params x 1.0625 B  (ASSUMPTION, see PLAN physics)
N_PREDICT_DEFAULT = 256
TOK_MARGIN = 64                 # prompt = L - n_predict - TOK_MARGIN, so a split slot of exactly L tokens holds it
LS = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072, "t8short": 1024}   # t8short: 8 clients vs np4, anchor for stack_topology.yaml:140-150 (T=8 full 124.56 tok/s)
FORK_BIN_DEFAULT = "/mnt/raid0/llm/llama.cpp-experimental/build-cpu/bin/llama-server"   # placeholder; pass --fork-binary

# ----------------------------------------------------------------------------------------------- argv
def prod_argv(np_, c, unified, mtp, port, threads, binary=None, extra=()):
    a = [str(binary or BIN), "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "-np", str(np_), "-c", str(c),
         "-t", str(threads), "-ub", "2048", "--flash-attn", "on", "--jinja", "-ctk", "q8_0", "-ctv", "q8_0",
         "--mlock", "--no-mmap", "--cache-ram", "32768", "--chat-template-file", TEMPLATE]
    if mtp:
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
    a += ["--reasoning", "off", "--device", "none", "--device-draft", "none"]
    a += ["--kv-unified" if unified else "--no-kv-unified"]
    a += list(extra)
    return a

def verify_against_launcher():
    """Re-derive the production frontdoor argv/env from the launcher (read-only) and diff against prod_argv()."""
    code = ("import sys,json;sys.path.insert(0,'.');sys.path.insert(0,'scripts/server')\n"
            "import scripts.server.orchestrator_stack as S\n"
            "rc=S.RegistryLoader().get_role('frontdoor');cmd=S.build_server_command(rc,8070,False,numa_instance=0)\n"
            "env=S.build_launch_env('frontdoor',{});bo,ld=S._stack_prior_runtime_overrides('frontdoor')\n"
            "S._apply_runtime_requirements_env(env,binary_override=bo,ld_paths=ld,preserve=S._role_env_overrides('frontdoor'))\n"
            "print(json.dumps({'cmd':cmd,'env':env,'prefix':S._numa_prefix('frontdoor',0)}))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ORCH, capture_output=True, text=True, timeout=180)
    live = json.loads(out.stdout.strip().splitlines()[-1])
    want = prod_argv(4, 262144, False, True, 8070, 96)[:-1]          # launcher emits no kv flag for frontdoor today
    got = [x for x in live["cmd"]]
    got[0] = os.path.realpath(got[0]); want[0] = os.path.realpath(want[0])
    # bench substitutions: drop --slot-save-path <dir>
    if "--slot-save-path" in got:
        i = got.index("--slot-save-path"); del got[i:i + 2]
    ok_argv = got == want
    ok_env = all(live["env"].get(k) == v for k, v in PROD_ENV.items())
    print(json.dumps({"argv_match": ok_argv, "env_match": ok_env, "prefix": live["prefix"]}))
    if not ok_argv:
        print("launcher :", got); print("driver   :", want)
    return ok_argv and ok_env

# ----------------------------------------------------------------------------------------------- matrix
def fits(np_, L, pool):             # every one of np concurrent streams needs L tokens
    return np_ * L <= pool

def build_launches(a):
    """Return list of launch dicts: {id, np, c, unified, mtp, cells:[(Lname, pmode)], kind}."""
    L_all = [k for k in LS if k in a.lens.split(",")]
    pmodes = a.pmodes.split(",")
    nps = [int(x) for x in a.nps.split(",")]
    out = []
    def add(lid, np_, c, uni, mtp, Ls, pm, kind="matrix", tier=None):
        base_pm = [x for x in pm if x != "shared_fork"]
        for arm_pm, arm_bin, arm_extra, suf in ((base_pm, None, (), ""),
                (["shared_fork"] if "shared_fork" in pm else [], a.fork_binary or FORK_BIN_DEFAULT,
                 ("--slot-fork-min-tokens", str(a.fork_min_tokens)), "_fork")):
            cells = [(l, p) for l in Ls for p in arm_pm]
            if cells:
                out.append(dict(id=lid + suf, np=np_, c=c, unified=uni, mtp=mtp, cells=cells, kind=kind,
                                bin=arm_bin, extra=tuple(a.extra_argv.split()) + tuple(arm_extra)))
    if "split" in a.kv_modes.split(","):
        for np_ in nps:
            add(f"split_np{np_}_slot65536", np_, np_ * 65536, False, True, [l for l in L_all if LS[l] <= 65536], pmodes)
            if np_ <= 4:
                add(f"split_np{np_}_slot131072", np_, np_ * 131072, False, True, [l for l in L_all if LS[l] == 131072], pmodes)
    if "unified" in a.kv_modes.split(","):
        for pool in [int(x) for x in a.pools.split(",")]:
            for np_ in nps:
                add(f"unified_np{np_}_pool{pool}", np_, pool, True, True, [l for l in L_all if fits(np_, LS[l], pool)], pmodes)
    if a.anchor:
        add("split_np4_slot65536_t8anchor", 4, 4 * 65536, False, True, ["t8short"], ["independent"], "anchor")
    if a.mtp_off:
        add("split_np8_slot65536_mtpoff", 8, 8 * 65536, False, False, ["64k"], ["independent"], "mtp_off")
        add("unified_np8_pool524288_mtpoff", 8, 524288, True, False, ["64k"], ["independent"], "mtp_off")
    if a.stress:
        add("unified_np8_pool524288_poolfull", 8, 524288, True, True, ["128k"], ["independent"], "poolfull")
    return out

def order_launches(ls, a):
    """Alternating A/B order (host drifts ~3%/few h): pair split/unified per np and flip the pair order on every
    other np; the drift anchor (split np4 L32k independent) is re-run at start, middle and end."""
    main = [l for l in ls if l["kind"] == "matrix"]
    by_np = {}
    for l in main:
        by_np.setdefault(l["np"], []).append(l)
    seq, flip = [], False
    for np_ in sorted(by_np):
        grp = sorted(by_np[np_], key=lambda l: (l["unified"], l["c"]))
        seq += (grp[::-1] if flip else grp)
        flip = not flip
    seq += [l for l in ls if l["kind"] != "matrix"]
    return seq

# ----------------------------------------------------------------------------------------------- estimator
def prefill_tps(L):               # ASSUMPTION calibrated to frontdoor live log (~298 tok/s @2.7k), mild L^2 attention tax
    return 300.0 * (1 - 0.35 * min(L, 131072) / 131072)
AGG_DECODE = {1: 40.0, 2: 70.0, 4: 100.0, 8: 124.0}     # tok/s MTP aggregate, frontdoor full host (registry/comments)
def est_cell_s(np_, Lname, pmode, waves, npred, mtp=True):
    L = LS[Lname]; P = L - npred - TOK_MARGIN
    agg = AGG_DECODE[np_] * (1.0 if mtp else 0.6)
    t_dec = npred / (agg / np_) * (1 + 0.25 * L / 131072)
    warm = P / prefill_tps(L) if pmode in ("shared", "shared_fork") else 0
    return warm + waves * (np_ * P / prefill_tps(L) + t_dec) + 20

def physics(np_, L, mtp=True):
    kv = np_ * L * KV_BYTES_PER_TOKEN / 1e9
    tok_per_step = np_ * (2.2 if mtp else 1.0)                # MTP verify batch ~ mean accepted+1 (ASSUMPTION)
    E, k = 256, 8                                              # experts/active (ASSUMPTION; PLAN says replace from GGUF)
    union = 1 - (1 - k / E) ** tok_per_step
    w = (W_ACTIVE1_GB * 0.2) + (W_TOTAL_GB - W_ACTIVE1_GB * 0.2) * union * 0.9   # 0.2 ~ non-expert share
    step_gb = kv + w
    steps_s = HOST_BW_GBS / step_gb
    return dict(kv_gb_step=round(kv, 2), weights_gb_step=round(w, 2), ceiling_steps_s=round(steps_s, 1),
                ceiling_agg_tok_s=round(steps_s * tok_per_step, 1))

# ----------------------------------------------------------------------------------------------- corpus / prompts
class Corpus:
    def __init__(self, roots, cap=24_000_000, seed=1):
        files = []
        for r in roots:
            files += sorted(Path(r).rglob("*.md"))
        files = [f for f in sorted(set(files)) if f.stat().st_size > 2048]
        blob, names = [], []
        n = 0
        for f in files:
            try:
                t = f.read_text(errors="ignore")
            except OSError:
                continue
            blob.append(t); names.append(str(f)); n += len(t)
            if n >= cap: break
        self.text = "\n\n".join(blob)
        self.sha = hashlib.sha256(self.text.encode()).hexdigest()
        self.nfiles = len(names)
        self.cursor = 0

    def take(self, nchars):
        if self.cursor + nchars > len(self.text):
            self.cursor = 0                                    # wrap (recorded as corpus_wrap in cell summary)
            self.wrapped = True
        s = self.text[self.cursor:self.cursor + nchars]
        self.cursor += nchars
        return s

def http(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def n_tokens(port, text):
    return len(http(port, "/tokenize", {"content": text, "add_special": False})["tokens"])

def make_doc(corpus, port, ntok, ratio):
    """Return (text, ntok_actual, ratio) of ~ntok tokens of fresh (never reused within a launch) corpus text."""
    nch = int(ntok * ratio)
    start = corpus.cursor
    for _ in range(4):
        corpus.cursor = start
        t = corpus.take(nch)
        n = n_tokens(port, t)
        if abs(n - ntok) <= max(8, ntok // 400):
            break
        nch = int(nch * ntok / max(n, 1))
    return t, n, len(t) / max(n, 1)

# ----------------------------------------------------------------------------------------------- one request
def one_request(port, prompt, npred, tag, t_barrier=None):
    body = {"prompt": prompt, "n_predict": npred, "ignore_eos": True, "temperature": 0, "seed": 42, "stream": True,
            "return_tokens": True, "cache_prompt": True}
    rec = {"tag": tag, "status": None, "error": None, "ttft": None, "t_start": None, "t_end": None, "chunks": []}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if t_barrier is not None:
        t_barrier.wait()
    t0 = time.time(); rec["t_start"] = t0
    last = {}
    try:
        with urllib.request.urlopen(req, timeout=14400) as r:
            rec["status"] = r.status
            for raw in r:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    if line.startswith("error"):
                        rec["error"] = line[:500]
                    continue
                try:
                    j = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if "error" in j:
                    rec["error"] = json.dumps(j["error"])[:500]; continue
                now = time.time()
                ntk = len(j.get("tokens") or []) or (1 if j.get("content") else 0)
                if ntk and rec["ttft"] is None:
                    rec["ttft"] = now - t0
                if ntk:
                    rec["chunks"].append((round(now - t0, 4), ntk))
                if j.get("stop"):
                    last = j
    except urllib.error.HTTPError as e:
        rec["status"] = e.code; rec["error"] = e.read().decode(errors="ignore")[:500]
    except Exception as e:                                      # noqa: BLE001 - recorded as evidence, not swallowed
        rec["error"] = repr(e)[:300]
    rec["t_end"] = time.time()
    tm = last.get("timings", {})
    rec.update(n_fork_tokens=tm.get("n_fork_tokens"), prompt_n=tm.get("prompt_n"), cache_n=tm.get("cache_n", last.get("tokens_cached")),
               prompt_ms=tm.get("prompt_ms"), prompt_tps=tm.get("prompt_per_second"),
               predicted_n=tm.get("predicted_n"), decode_tps=tm.get("predicted_per_second"),
               tokens_evaluated=last.get("tokens_evaluated"), tokens_cached=last.get("tokens_cached"),
               truncated=last.get("truncated"), stop_type=last.get("stop_type"))
    return rec

def overlap_agg(recs):
    """Aggregate decode tok/s inside the window where EVERY stream is decoding (max first-token .. min end)."""
    ok = [r for r in recs if r["ttft"] is not None and r["chunks"]]
    if len(ok) < len(recs) or not ok:
        return None
    lo = max(r["t_start"] + r["ttft"] for r in ok); hi = min(r["t_end"] for r in ok)
    if hi - lo < 1.0:
        return None
    n = 0
    for r in ok:
        for t, k in r["chunks"]:
            if lo <= r["t_start"] + t <= hi:
                n += k
    return n / (hi - lo)

# ----------------------------------------------------------------------------------------------- server mgmt
class Server:
    def __init__(self, argv, env, prefix, logdir, port):
        self.argv, self.env, self.prefix, self.dir, self.port = argv, env, prefix, logdir, port
        self.proc = None

    def start(self):
        self.out = open(self.dir / "server.log", "ab")
        self.proc = subprocess.Popen(self.prefix + self.argv, stdout=self.out, stderr=subprocess.STDOUT, env=self.env)
        (self.dir / "server.pid").write_text(str(self.proc.pid))
        PIDS.add(self.proc.pid)

    def healthy(self, timeout=900):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.proc.poll() is not None:
                return "DIED"
            try:
                if http(self.port, "/health", timeout=3).get("status") == "ok":
                    return "HEALTHY"
            except Exception:
                pass
            time.sleep(3)
        return "TIMEOUT"

    def rss_gib(self):
        try:
            d = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{self.proc.pid}/status") if l.startswith(("VmRSS", "VmHWM"))}
            return round(d["VmRSS"] / 1048576, 2), round(d["VmHWM"] / 1048576, 2)
        except Exception:
            return None, None

    def linkage(self):
        libs = set()
        try:
            for l in open(f"/proc/{self.proc.pid}/maps"):
                p = l.split()[-1]
                if re.search(r"lib(ggml|llama)[^/]*\.so", p):
                    libs.add(os.path.dirname(os.path.realpath(p)))
        except Exception as e:
            return {"error": repr(e)}
        want = getattr(self, 'bin_dir', None) or os.path.realpath(KSTORE)
        return {"lib_dirs": sorted(libs), "all_in_kernel_store": libs == {want} or (bool(libs) and all(x == want for x in libs))}

    def stop(self):
        pid = self.proc.pid if self.proc else None
        if pid is None:
            return True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait(30)
        dead = not alive(pid)
        if dead:
            PIDS.discard(pid)
        return dead

PIDS = set()
def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                          # zombie counts as dead
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False

def _cleanup(*_):
    for pid in list(PIDS):                                        # only PIDs this driver started
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 30
    while time.time() < deadline and any(alive(p) for p in PIDS):
        time.sleep(1)
    for pid in list(PIDS):
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
atexit.register(_cleanup)
for _s in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_s, lambda *x: (_cleanup(), sys.exit(130)))

LOAD_PATS = {"kv_unified": r"kv_unified = '(\w+)'", "n_ctx_slot": r"n_ctx_slot = (\d+)", "n_slots": r"n_slots = (\d+)",
             "kv_buffer_mib": r"KV buffer size =\s*([\d.]+) MiB", "rs_buffer_mib": r"(?:RS|rs|recurrent)[^\n]*?(?:buffer )?size =\s*([\d.]+) MiB",
             "n_ctx_train_warn": r"(n_ctx_seq[^\n]*n_ctx_train[^\n]*)"}
ERR_PATS = ["sub-batch", "exception", "failed to", "context size", "exceed", r"truncated = [1-9]", "error", "assert", "OOM", "out of memory"]

def scan_log(path, offset):
    data = open(path, "rb").read()[offset:].decode(errors="ignore")
    return {p: len(re.findall(p, data, re.I)) for p in ERR_PATS if re.search(p, data, re.I)}, len(data.encode()) + offset

# ----------------------------------------------------------------------------------------------- cells
def run_launch(a, la, outroot, corpus, summary):
    ldir = outroot / la["id"]; ldir.mkdir(parents=True, exist_ok=True)
    argv = prod_argv(la["np"], la["c"], la["unified"], la["mtp"], a.port, a.threads, la.get("bin") or a.binary, la.get("extra", ()))
    env = dict(os.environ); env.update(PROD_ENV)
    env.pop("GGML_NUMA_REPACK_INTERLEAVE", None)
    prefix = ["numactl", "--interleave=all", "--", "taskset", "-c", a.cpu_list]
    log = open(ldir / "launch.log", "a")
    def say(s):
        line = f"[{time.strftime('%FT%TZ', time.gmtime())}] {s}"; print(line, flush=True); log.write(line + "\n"); log.flush()
    say(f"LAUNCH {la['id']} argv={json.dumps(prefix + argv)}")
    say(f"ENV {json.dumps({k: env.get(k) for k in PROD_ENV})} binary={os.path.realpath(argv[0])}")
    srv = Server(argv, env, prefix, ldir, a.port); srv.bin_dir = os.path.dirname(os.path.realpath(argv[0])); srv.start()
    try:
        st = srv.healthy()
        say(f"health={st}")
        if st != "HEALTHY":
            summary(dict(launch=la["id"], cell="-", result=f"SERVER_{st}")); return
        link = srv.linkage(); say(f"LINKAGE {json.dumps(link)}")
        if not link.get("all_in_kernel_store"):
            summary(dict(launch=la["id"], cell="-", result="LINKAGE_MISMATCH", linkage=link)); return
        slog = (ldir / "server.log").read_text(errors="ignore")
        facts = {k: (re.findall(p, slog) or [None])[-1] for k, p in LOAD_PATS.items()}
        rss0 = srv.rss_gib(); facts.update(rss_after_load_gib=rss0[0])
        say(f"LOADFACTS {json.dumps(facts)}")
        (ldir / "loadfacts.json").write_text(json.dumps(dict(facts=facts, linkage=link, argv=prefix + argv), indent=1))
        if a.loadonly:
            summary(dict(launch=la["id"], cell="loadonly", result="OK", **facts)); return
        # warm-up: one short untimed request (graph build, page-in), not counted
        one_request(a.port, "Warm up. " * 50, 16, "warmup")
        off = len(open(ldir / "server.log", "rb").read())
        ratio = 4.0
        for (Lname, pmode) in la["cells"]:
            if la["kind"] == "poolfull":
                run_poolfull(a, la, ldir, corpus, Lname, srv, summary, say, facts); continue
            ratio = run_cell(a, la, ldir, corpus, Lname, pmode, srv, summary, say, facts, ratio, off)
            off = len(open(ldir / "server.log", "rb").read())
    finally:
        dead = srv.stop(); say(f"STOP dead={dead} pid={srv.proc.pid}")
        if not dead:
            summary(dict(launch=la["id"], cell="-", result="KILL_FAILED", pid=srv.proc.pid))

def cell_header(cl, a, la, Lname, pmode, srv, facts):
    cl.write(f"CELL_CONFIG launch={la['id']} L={Lname} pmode={pmode} np={la['np']} c={la['c']} unified={la['unified']} "
             f"mtp={la['mtp']} waves={a.waves} n_predict={a.npredict}\nARGV {json.dumps(srv.prefix + srv.argv)}\n"
             f"ENV {json.dumps({k: srv.env.get(k) for k in PROD_ENV})}\nBINARY {os.path.realpath(srv.argv[0])} LINKAGE {json.dumps(srv.linkage())}\n"
             f"LOADFACTS {json.dumps(facts)}\nPHYSICS {json.dumps(physics(la['np'], LS[Lname], la['mtp']))}\n")

def run_cell(a, la, ldir, corpus, Lname, pmode, srv, summary, say, facts, ratio, off):
    L, np_ = LS[Lname], la["np"]
    nclients = 8 if Lname == "t8short" else np_
    ntok = L - a.npredict - TOK_MARGIN
    cdir = ldir / f"{Lname}_{pmode}"; cdir.mkdir(exist_ok=True)
    cl = open(cdir / "cell.log", "w"); cell_header(cl, a, la, Lname, pmode, srv, facts)
    waves = []
    t_cell = time.time()
    shared_doc = None
    for w in range(a.waves):
        prompts = []
        if pmode in ("shared", "shared_fork"):
            if shared_doc is None:
                shared_doc, n, ratio = make_doc(corpus, a.port, ntok - 40, ratio)
                warm = one_request(a.port, shared_doc + "\n\n[warm]\n", 1, "warm")
                cl.write(f"SHARED_WARM prompt_n={warm['prompt_n']} cache_n={warm['cache_n']} err={warm['error']}\n")
            for s in range(nclients):
                prompts.append(shared_doc + f"\n\n[stream {s} wave {w} nonce {hashlib.sha1(f'{la['id']}{Lname}{w}{s}'.encode()).hexdigest()[:8]}] "
                               "Summarize the document above in detail.\n")
        else:
            for s in range(nclients):
                d, n, ratio = make_doc(corpus, a.port, ntok - 40, ratio)
                prompts.append(d + f"\n\n[stream {s} wave {w} nonce {hashlib.sha1(f'{la['id']}{Lname}{w}{s}'.encode()).hexdigest()[:8]}] "
                               "Summarize the document above in detail.\n")
        bar = threading.Barrier(nclients)
        t0 = time.time()
        with ThreadPoolExecutor(nclients) as ex:
            recs = list(ex.map(lambda sp: one_request(a.port, sp[1], a.npredict, f"w{w}s{sp[0]}", bar), enumerate(prompts)))
        wall = time.time() - t0
        rss, hwm = srv.rss_gib()
        ok = [r for r in recs if r["error"] is None and r["status"] == 200]
        dec = [r["decode_tps"] for r in ok if r["decode_tps"]]
        wv = dict(wave=w, wall_s=round(wall, 1), n_ok=len(ok), n_err=len(recs) - len(ok),
                  ttft_s=[round(r["ttft"], 2) if r["ttft"] else None for r in recs],
                  prefill_tps=[round(r["prompt_tps"], 1) if r["prompt_tps"] else None for r in recs],
                  prompt_n=[r["prompt_n"] for r in recs], n_fork_tokens=[r["n_fork_tokens"] for r in recs], n_prompt_tokens_cache=[r["cache_n"] for r in recs],
                  predicted_n=[r["predicted_n"] for r in recs], decode_tps_per_stream=[round(x, 2) for x in dec],
                  decode_tps_sum=round(sum(dec), 2), decode_tps_overlap_window=(round(overlap_agg(recs), 2) if overlap_agg(recs) else None),
                  agg_e2e_tok_s=round(sum(r["predicted_n"] or 0 for r in ok) / wall, 2),
                  rss_gib=rss, vmhwm_gib=hwm, errors=[r["error"] for r in recs if r["error"]][:4])
        waves.append(wv)
        cl.write("WAVE " + json.dumps(wv) + "\n"); cl.flush()
        say(f"{la['id']} {Lname} {pmode} wave{w}: wall={wv['wall_s']}s ttft={wv['ttft_s']} dec_sum={wv['decode_tps_sum']} cache={wv['n_prompt_tokens_cache']} err={wv['n_err']}")
    errs, _ = scan_log(ldir / "server.log", off)
    def med(key):
        v = [x for w in waves for x in (w[key] if isinstance(w[key], list) else [w[key]]) if x is not None]
        return round(statistics.median(v), 2) if v else None
    row = dict(launch=la["id"], cell=f"{Lname}_{pmode}", np=np_, clients=nclients, n_fork_tokens_by_wave=[w["n_fork_tokens"] for w in waves], prompt_n_by_wave=[w["prompt_n"] for w in waves], c=la["c"], unified=la["unified"], mtp=la["mtp"], L=L, pmode=pmode,
               waves=a.waves, result="OK" if all(w["n_err"] == 0 for w in waves) else "ERRORS",
               prefill_tps_med=med("prefill_tps"), ttft_s_med=med("ttft_s"), decode_tps_stream_med=med("decode_tps_per_stream"),
               decode_tps_sum_med=med("decode_tps_sum"), decode_tps_overlap_med=med("decode_tps_overlap_window"),
               agg_e2e_med=med("agg_e2e_tok_s"), cache_n_by_wave=[w["n_prompt_tokens_cache"] for w in waves],
               rss_gib_max=max((w["rss_gib"] or 0) for w in waves), stderr_hits=errs, cell_wall_s=round(time.time() - t_cell, 1),
               physics=physics(np_, L, la["mtp"]), host_bw_gbs=HOST_BW_GBS, n_ctx_slot=facts.get("n_ctx_slot"), kv_unified=facts.get("kv_unified"))
    cl.write("CELL_SUMMARY " + json.dumps(row) + "\n"); cl.close()
    (cdir / "waves.json").write_text(json.dumps(waves, indent=1))
    summary(row)
    return ratio

def run_poolfull(a, la, ldir, corpus, Lname, srv, summary, say, facts):
    """Deliberately oversubscribe the unified pool: np streams x L tokens > pool.  Record error vs queue vs truncate."""
    L, np_ = LS[Lname], la["np"]
    ntok = L - a.npredict - TOK_MARGIN
    cdir = ldir / f"{Lname}_poolfull"; cdir.mkdir(exist_ok=True)
    cl = open(cdir / "cell.log", "w"); cell_header(cl, a, la, Lname, "independent", srv, facts)
    cl.write(f"POOLFULL demand={np_ * L} pool={la['c']} oversub={np_ * L / la['c']:.2f}x\n")
    prompts, ratio = [], 4.0
    for s in range(np_):
        d, n, ratio = make_doc(corpus, a.port, ntok - 40, ratio)
        prompts.append(d + f"\n\n[stream {s} poolfull] Summarize the document above in detail.\n")
    stop_poll = threading.Event(); polls = []
    def poll():
        while not stop_poll.is_set():
            try:
                sl = http(a.port, "/slots", timeout=5)
                polls.append(dict(t=round(time.time() - t0, 1), slots=[dict(id=s.get("id"), proc=s.get("is_processing"), n_ctx=s.get("n_ctx"),
                                  n_past=(s.get("next_token") or [{}])[0].get("n_decoded")) for s in sl]))
            except Exception as e:
                polls.append(dict(t=round(time.time() - t0, 1), err=repr(e)[:80]))
            stop_poll.wait(15)
    t0 = time.time(); th = threading.Thread(target=poll, daemon=True); th.start()
    off = len(open(ldir / "server.log", "rb").read())
    bar = threading.Barrier(np_)
    with ThreadPoolExecutor(np_) as ex:
        recs = list(ex.map(lambda sp: one_request(a.port, sp[1], a.npredict, f"pf{sp[0]}", bar), enumerate(prompts)))
    stop_poll.set(); th.join(10)
    errs, _ = scan_log(ldir / "server.log", off)
    per = [dict(tag=r["tag"], status=r["status"], error=r["error"], ttft=r["ttft"], prompt_n=r["prompt_n"], predicted_n=r["predicted_n"],
                truncated=r["truncated"], stop_type=r["stop_type"], wall=round(r["t_end"] - r["t_start"], 1)) for r in recs]
    outcome = ("ERROR" if any(r["status"] != 200 or r["error"] for r in recs) else
               "TRUNCATE" if any(r["truncated"] for r in recs) else "QUEUE_OR_FIT")
    row = dict(launch=la["id"], cell=f"{Lname}_poolfull", np=np_, c=la["c"], L=L, result="POOLFULL", outcome=outcome, per_request=per,
               stderr_hits=errs, slots_polls=polls[:200], rss_gib_max=srv.rss_gib()[1])
    cl.write("CELL_SUMMARY " + json.dumps(row) + "\n"); cl.close()
    summary(row)

# ----------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print every launch+cell argv and estimates; launch nothing")
    p.add_argument("--verify-against-launcher", action="store_true")
    p.add_argument("--out", default=str(ART / "runs"))
    p.add_argument("--kv-modes", default="split,unified")
    p.add_argument("--nps", default="1,2,4,8")
    p.add_argument("--lens", default="8k,32k,64k,128k")
    p.add_argument("--pmodes", default="independent,shared", help="add shared_fork (needs --fork-binary at run time; dry-run prints a placeholder) for the prefix-fork arm")
    p.add_argument("--pools", default="262144,524288,1048576")
    p.add_argument("--waves", type=int, default=3)
    p.add_argument("--npredict", type=int, default=N_PREDICT_DEFAULT)
    p.add_argument("--mtp-off", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--anchor", action="store_true", help="add the T=8-clients/np4 short-ctx cell comparable to stack_topology.yaml:140-150")
    p.add_argument("--binary", default=str(BIN), help="server binary for the plain arms (default: production v10 kernel store)")
    p.add_argument("--extra-argv", default="", help="extra server argv for ALL arms (space separated)")
    p.add_argument("--fork-binary", default="", help="cross-slot prefix-fork build; enables the shared_fork prompt-mode arm")
    p.add_argument("--fork-min-tokens", type=int, default=1024, help="value for --slot-fork-min-tokens in the shared_fork arm (0 = off)")
    p.add_argument("--only", default="", help="comma list of launch ids (substring match)")
    p.add_argument("--cpu-list", default="0-95")
    p.add_argument("--threads", type=int, default=96)
    p.add_argument("--port", type=int, default=18970)
    p.add_argument("--loadonly", action="store_true", help="launch, record load facts, stop (correctness-class)")
    p.add_argument("--corpus-roots", default="/workspace/handoffs,/workspace/docs")
    a = p.parse_args()
    if a.port in range(8000, 8201):
        sys.exit("refusing production port range 8000-8200")
    if a.verify_against_launcher:
        sys.exit(0 if verify_against_launcher() else 1)
    ls = order_launches(build_launches(a), a)
    if a.only:
        keys = a.only.split(","); ls = [l for l in ls if any(k in l["id"] for k in keys)]
    if a.dry_run:
        tot, ncell = 0.0, 0
        for la in ls:
            argv = prod_argv(la["np"], la["c"], la["unified"], la["mtp"], a.port, a.threads, la.get("bin") or a.binary, la.get("extra", ()))
            print(f"\n== launch {la['id']} kind={la['kind']} np={la['np']} c={la['c']} unified={la['unified']} mtp={la['mtp']}")
            print("   ARGV " + " ".join(["numactl", "--interleave=all", "--", "taskset", "-c", a.cpu_list] + argv))
            print("   ENV  " + " ".join(f"{k}={v}" for k, v in PROD_ENV.items()))
            tot += 120
            for (l, pm) in la["cells"]:
                e = est_cell_s(la["np"], l, pm, a.waves, a.npredict, la["mtp"]) if la["kind"] != "poolfull" else la["np"] * (LS[l] / prefill_tps(LS[l]))
                tot += e; ncell += 1
                print(f"   cell {l:>4} {pm:<11} est {e/60:7.1f} min  physics {physics(la['np'], LS[l], la['mtp'])}")
        print(f"\nTOTAL launches={len(ls)} cells={ncell} est={tot/3600:.1f} h (estimator is an ASSUMPTION model; smoke calibrates prefill_tps)")
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(a.corpus_roots.split(","))
    sumf = out / "summary.jsonl"
    def summary(row):
        row["ts"] = time.strftime("%FT%TZ", time.gmtime())
        with open(sumf, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("SUMMARY " + json.dumps({k: row.get(k) for k in ("launch", "cell", "result", "prefill_tps_med", "ttft_s_med", "decode_tps_sum_med", "agg_e2e_med")}), flush=True)
    print(f"corpus sha256={corpus.sha} files={corpus.nfiles} chars={len(corpus.text)}", flush=True)
    r = subprocess.run([LINKAGE, str(BIN)], capture_output=True, text=True)
    print("verify_ggml_linkage rc=%d\n%s%s" % (r.returncode, r.stdout[-600:], r.stderr[-300:]), flush=True)
    if r.returncode != 0:
        sys.exit("verify_ggml_linkage failed; refusing to measure")
    for b in sorted({la.get("bin") for la in ls if la.get("bin")}):
        rb = subprocess.run([LINKAGE, b], capture_output=True, text=True)
        print(f"verify_ggml_linkage {b} rc={rb.returncode}", flush=True)
        if rb.returncode != 0:
            sys.exit("fork binary linkage failed")
    for la in ls:
        run_launch(a, la, out, corpus, summary)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
