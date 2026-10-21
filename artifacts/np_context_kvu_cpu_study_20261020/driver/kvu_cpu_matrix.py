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
DEPTH_SUFFIX_TOK = 256          # tier-0 depth mode: fresh suffix per wave (<=256 tokens)
TOK_MARGIN = 64                 # prompt = L - n_predict - TOK_MARGIN, so a split slot of exactly L tokens holds it
# KVU-12.4 slot-fit budget (proposed-by-workspace-f8 2026-10-21; origin: W2 2026-10-21, the 4th stream's ~65.5k-token request exceeded the
# 65,536 split slot -> HTTP 400 in every split decode wave).  Depth-mode document tokens are budgeted from the slot, not from L-n_predict:
#   doc = slot - SLOT_MARGIN_TOK - n_predict - DEPTH_SUFFIX_RESERVE_TOK - TEMPLATE_OVERHEAD_TOK
SLOT_MARGIN_TOK = 256           # explicit planning margin kept free in every slot
DEPTH_SUFFIX_RESERVE_TOK = 384  # header + <=256-token suffix + ratio slack (W2 measured suffix turns of 261-310 tokens incl. header)
TEMPLATE_OVERHEAD_TOK = 64      # BOS / template / tokenizer-boundary slack (the cells use raw prompts, so this is conservative)
GUARD_HEADROOM_TOK = 32         # HARD refusal: planned request tokens + n_predict must be <= slot - this
LS = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072, "t8short": 1024}   # t8short: 8 clients vs np4, anchor for stack_topology.yaml:140-150 (T=8 full 124.56 tok/s)
FORK_BIN_DEFAULT = "/mnt/raid0/llm/llama.cpp-experimental/build-cpu/bin/llama-server"   # placeholder; pass --fork-binary
BANK_NAMES = ["8k", "32k", "64k", "128k"]
BANK_DOCS = 8
def depth_doc_tok(L, npredict):
    """Tokens of the per-stream depth-mode document so that wave 0 AND every suffix turn fit a slot of L tokens."""
    return L - SLOT_MARGIN_TOK - npredict - DEPTH_SUFFIX_RESERVE_TOK - TEMPLATE_OVERHEAD_TOK

class CellRefused(Exception):
    """Raised BEFORE any request is sent when a planned request would not fit the per-stream slot (never a silent HTTP 400)."""

def slot_limit(la, L):
    """Tokens one stream may occupy: min(L, per-slot ctx in split mode).  Unified arms are held to the same L so both arms carry equal streams."""
    return L if la["unified"] else min(L, la["c"] // la["np"])

def bank_tok(name):                       # tokens in the bank doc for context class `name` (leaves 512 for suffix + 256 gen)
    return LS[name] - 512

# ----------------------------------------------------------------------------------------------- argv
def prod_argv(np_, c, unified, mtp, port, threads, binary=None, extra=(), no_cis=True):
    a = [str(binary or BIN), "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "-np", str(np_), "-c", str(c),
         "-t", str(threads), "-ub", "2048", "--flash-attn", "on", "--jinja", "-ctk", "q8_0", "-ctv", "q8_0",
         "--mlock", "--no-mmap", "--cache-ram", "32768", "--chat-template-file", TEMPLATE]
    if mtp:
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
    a += ["--reasoning", "off", "--device", "none", "--device-draft", "none"]
    a += ["--kv-unified" if unified else "--no-kv-unified"]
    # KVU-12.4: BOTH arms carry --no-cache-idle-slots (W2 ran it on unified only, a confound); the only argv difference between the arms
    # is the KV mode under test (--kv-unified/--no-kv-unified and the -c that geometry implies).  no_cis=False only for verify_against_launcher.
    if no_cis and "--no-cache-idle-slots" not in extra:
        a += ["--no-cache-idle-slots"]
    a += list(extra)
    return a

def norm_argv(argv):
    """argv with ONLY the KV-mode-under-test tokens removed (-c geometry, --kv-unified/--no-kv-unified) plus run-local noise (port, slot-save-path)."""
    out, i = [], 0
    while i < len(argv):
        x = argv[i]
        if x in ("-c", "--port", "--slot-save-path"):
            i += 2; continue
        if x in ("--kv-unified", "--no-kv-unified"):
            i += 1; continue
        out.append(x); i += 1
    return out

def aba_guard(a, ls):
    """KVU-12.4 argv-diff guard: both arms of an A/B must be identical except the KV mode under test.  Refuses (exit 4) a mismatched ABA, in-process
    (launches that share np + cell set) and across driver invocations via --aba-ledger/--aba-key (the window launcher runs one launch per invocation)."""
    def argv_of(la):
        return prod_argv(la["np"], la["c"], la["unified"], la["mtp"], a.port, a.threads, la.get("bin") or a.binary, la.get("extra", ()))
    cand = [la for la in ls if la["kind"] in ("matrix", "mtp_off") and any(pm == "depth" for _, pm in la["cells"])]
    bad = []
    for la in cand:
        if "--no-cache-idle-slots" not in argv_of(la):
            bad.append(f"{la['id']}: --no-cache-idle-slots missing")
    groups = {}
    for la in cand:
        groups.setdefault((la["np"], la["mtp"], tuple(sorted({l for l, _ in la["cells"]}))), []).append(la)
    for key, g in groups.items():
        base = norm_argv(argv_of(g[0]))
        for la in g[1:]:
            n2 = norm_argv(argv_of(la))
            if n2 != base:
                bad.append(f"{g[0]['id']} vs {la['id']}: argv differs beyond KV mode: {[x for x in n2 if x not in base]} / {[x for x in base if x not in n2]}")
    if a.aba_key and a.aba_ledger:
        led = Path(a.aba_ledger)
        prior = [json.loads(x) for x in led.read_text().splitlines() if x.strip()] if led.exists() else []
        for la in cand:
            n = norm_argv(argv_of(la)); n[0] = os.path.realpath(n[0])
            for pr in prior:
                if pr["key"] == a.aba_key and pr["norm_argv"] != n:
                    bad.append(f"ledger {a.aba_ledger} key {a.aba_key}: {la['id']} differs from {pr['launch']}: "
                               f"+{[x for x in n if x not in pr['norm_argv']]} -{[x for x in pr['norm_argv'] if x not in n]}")
        if not bad and not a.dry_run:
            with open(led, "a") as f:
                for la in cand:
                    n = norm_argv(argv_of(la)); n[0] = os.path.realpath(n[0])
                    f.write(json.dumps(dict(key=a.aba_key, launch=la["id"], norm_argv=n, ts=time.strftime("%FT%TZ", time.gmtime()))) + "\n")
    if bad:
        print("ABA_ARGV_GUARD REFUSED:\n  " + "\n  ".join(bad), flush=True)
        sys.exit(4)
    print(f"ABA_ARGV_GUARD ok: {len(cand)} depth launches, groups={len(groups)}, ledger={'yes' if a.aba_key else 'no'}", flush=True)

def static_fit_check(a, ls):
    """Arithmetic refusal before anything launches: planned worst-case request (doc budget + suffix reserve + template) + n_predict must fit slot - guard."""
    bad = []
    for la in ls:
        for (l, pm) in la["cells"]:
            if pm == "depth" and l in LS:
                lim = slot_limit(la, LS[l])
                worst = depth_doc_tok(LS[l], a.npredict) + DEPTH_SUFFIX_RESERVE_TOK + TEMPLATE_OVERHEAD_TOK + a.npredict
                if worst > lim - GUARD_HEADROOM_TOK:
                    bad.append(f"{la['id']} {l}: planned {worst} > slot {lim} - guard {GUARD_HEADROOM_TOK}")
                elif a.dry_run:
                    print(f"   SLOT_FIT {la['id']} {l} depth: doc {depth_doc_tok(LS[l], a.npredict)} + suffix<= {DEPTH_SUFFIX_RESERVE_TOK} + tmpl {TEMPLATE_OVERHEAD_TOK} + npredict {a.npredict} = {worst} <= slot {lim} (headroom {lim - worst})")
    if bad:
        print("SLOT_FIT_STATIC REFUSED:\n  " + "\n  ".join(bad), flush=True); sys.exit(5)

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
    want = prod_argv(4, 262144, False, True, 8070, 96, no_cis=False)[:-1]          # launcher emits no kv flag for frontdoor today
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

def build_tier0(a):
    """Tier 0 decision core on the CONTEXT BANK (fixed order, ABA where a split/unified pair exists).  Every bank cell restores
    doc{i}_L{L}.bin and decodes; the only live long prefills left are the shared-prefix / fork cells (that IS their subject)."""
    mk = lambda lid, np_, c, uni, mtp, cells, kind="matrix", **kw: dict(id=lid, np=np_, c=c, unified=uni, mtp=mtp, cells=cells, kind=kind, bin=kw.get("bin"),
                                                                        extra=tuple(a.extra_argv.split()) + ("--slot-save-path", bank_sub(a.bank_dir, uni, np_)) + tuple(kw.get("extra", ())))
    A = lambda: mk("split_np4_slot65536", 4, 262144, False, True, [("64k", "bank"), ("t8short", "independent")])
    out = [A(),                                                                              # A: first cell also refreshes decode calibration
           mk("unified_np4_pool524288", 4, 524288, True, True, [("64k", "bank")]),            # B
           dict(A(), id="split_np4_slot65536_aba2", cells=[("64k", "bank")]),                 # A again (drift)
           mk("unified_np8_pool524288", 8, 524288, True, True, [("32k", "bank"), ("64k", "bank"), ("64k", "bank4"), ("64k", "bank_resident4"), ("64k", "shared")]),
           mk("unified_np8_pool524288_mtpoff", 8, 524288, True, False, [("64k", "bank")], "mtp_off"),
           mk("unified_np8_pool1048576", 8, 1048576, True, True, [("128k", "bank")]),
           mk("unified_np8_pool524288_poolfull", 8, 524288, True, True, [("128k", "poolfull_bank")], "poolfull_bank")]
    if a.fork_binary or a.dry_run:
        out.append(mk("unified_np8_pool524288_fork", 8, 524288, True, True, [("64k", "shared_fork")], bin=a.fork_binary or FORK_BIN_DEFAULT,
                      extra=("--slot-fork-min-tokens", str(a.fork_min_tokens))))
    return out

def est_bank_build(npr=BANK_DOCS, names=None):
    """Seconds to build the bank once: each chunk is prefilled at its end depth's rate; plus saves/hash (~2 min/doc) and controls."""
    t, prev = 0.0, 0
    names = names or BANK_NAMES
    for nm in names:
        D = bank_tok(nm); t += (D - prev) / prefill_tps(D); prev = D
    ctl = sum(bank_tok(nm) / prefill_tps(bank_tok(nm)) for nm in names)
    return npr * (t + 120) + ctl + 180

def est_bank_cell(np_, Lname, pmode, waves, npred, mtp=True):
    active, nrest = parse_bank_mode(pmode, np_)
    L = LS[Lname]
    agg = AGG_DECODE[min(active, 8) if active in AGG_DECODE else 8] * (1.0 if mtp else 0.6)
    t_dec = npred / (agg / active) * (1 + 0.25 * L / 131072)
    t_rest = nrest * (L * KV_BYTES_PER_TOKEN / 1.5e9 + 1.0)              # file read from page cache + state load, untimed but wall
    t_sfx = active * 200 / prefill_tps(L)
    return waves * (t_rest + t_sfx + t_dec) + 60

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
    if pmode in ("depth", "shared", "shared_fork"):
        agg = AGG_DECODE[np_] * (1.0 if mtp else 0.6)
        t_dec = npred / (agg / np_) * (1 + 0.25 * L / 131072)
        re_pre = np_ * (2048 + DEPTH_SUFFIX_TOK) / prefill_tps(L)      # checkpoint-granularity re-process + suffix
        first = {"depth": np_ * P / prefill_tps(L),
                 "shared": P / prefill_tps(L) + (np_ - 1) * P / prefill_tps(L),      # warm + np-1 cold streams
                 "shared_fork": P / prefill_tps(L) + re_pre}[pmode]                  # warm + forked streams
        return first + t_dec + (waves - 1) * (re_pre + t_dec) + 20
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
    for _ in range(8):        # KVU-12.4: a document must never exceed its token budget (W2 docs overshot by up to ~380 tokens)
        if n <= ntok:
            break
        t = t[:max(1, len(t) - int((n - ntok) * len(t) / max(n, 1) * 1.05) - 1)]
        n = n_tokens(port, t)
    if n > ntok:
        raise CellRefused(f"make_doc could not fit {n} tokens into budget {ntok}")
    return t, n, len(t) / max(n, 1)

# ----------------------------------------------------------------------------------------------- one request
def one_request(port, prompt, npred, tag, t_barrier=None, id_slot=None, stream=True, timeout=14400):
    body = {"prompt": prompt, "n_predict": npred, "ignore_eos": True, "temperature": 0, "seed": 42, "stream": stream,
            "return_tokens": True, "cache_prompt": True}
    if id_slot is not None:
        body["id_slot"] = id_slot          # pin stream -> slot so per-slot reuse is deterministic (depth mode)
    rec = {"tag": tag, "status": None, "error": None, "ttft": None, "t_start": None, "t_end": None, "chunks": [], "text": ""}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if t_barrier is not None:
        t_barrier.wait()
    t0 = time.time(); rec["t_start"] = t0
    last = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec["status"] = r.status
            if not stream:
                last = json.loads(r.read()); rec["text"] = last.get("content", ""); rec["ttft"] = time.time() - t0
            for raw in (r if stream else []):
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
                rec["text"] += j.get("content") or ""
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
    rec.update(draft_n=tm.get("draft_n"), draft_n_accepted=tm.get("draft_n_accepted"), n_fork_tokens=tm.get("n_fork_tokens"), prompt_n=tm.get("prompt_n"), cache_n=tm.get("cache_n", last.get("tokens_cached")),
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

# ----------------------------------------------------------------------------------------------- capture rows
# Belief-kernel WRITE side (CLAUDE.md "Belief Kernel"; VB-KVU-1 / producer for VB-KVU-CPU).  Projection only: no grading.
CAPTURE_SCHEMA = "epyc.vidya.np_ctx_kvu_capture.v1"
_BIN_SHA = {}
def file_sha256(path):
    if path not in _BIN_SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        _BIN_SHA[path] = h.hexdigest()
    return _BIN_SHA[path]

def row_hash(row):
    body = {k: v for k, v in row.items() if k != "row_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def validate_row(row):
    """Shared validator (the adapter imports this).  Returns a list of problems; [] = valid."""
    bad = []
    if row.get("schema") != CAPTURE_SCHEMA: bad.append("schema")
    if row.get("row_sha256") != row_hash(row): bad.append("row_sha256 mismatch")
    if row.get("kv_unified_from_log") not in ("true", "false"): bad.append("kv_unified not read back from launch log")
    for k in ("binary_sha256", "binary_realpath", "n_ctx_slot", "c", "np", "prompt_mode", "ts", "cpu_list", "loadavg_start"):
        if row.get(k) in (None, ""): bad.append(f"missing {k}")
    return bad

def finalize_row(row, a, la, srv, facts, loadavg, wave=None, wave_rec=None):
    """Stamp a cell (wave=None) or per-wave row with the capture schema + self-hash.  Fields come from the launch
    log / filesystem / the request records, never from argv defaults."""
    argv = srv.argv
    binp = os.path.realpath(argv[0])
    dd = argv[argv.index("--spec-draft-n-max") + 1] if "--spec-draft-n-max" in argv else None
    row = dict(row)
    row.update(schema=CAPTURE_SCHEMA, model=os.path.basename(MODEL), arm=la["id"], wave=wave,
               prompt_mode=row.get("pmode") or row.get("prompt_mode") or "independent",
               binary_realpath=binp, binary_sha256=file_sha256(binp), kernel_store_path=os.path.realpath(KSTORE),
               kv_unified_from_log=facts.get("kv_unified"), cache_idle_slots_argv=("off" if "--no-cache-idle-slots" in argv else "on(default)"), extra_argv=[x for x in argv[argv.index("--no-kv-unified" if "--no-kv-unified" in argv else "--kv-unified") + 1:]], kv_unified_argv_flag=("--kv-unified" if "--kv-unified" in argv else "--no-kv-unified" if "--no-kv-unified" in argv else None),
               n_ctx_slot=int(facts["n_ctx_slot"]) if facts.get("n_ctx_slot") else None, n_slots_log=facts.get("n_slots"),
               n_ctx_train_warn=facts.get("n_ctx_train_warn"),
               pool=la["c"] if la["unified"] else None, c=la["c"], np=la["np"], L=row.get("L", LS.get(la["cells"][0][0]) if la.get("cells") else None),
               mtp=la["mtp"], draft_depth=(int(dd) if dd and la["mtp"] else 0), draft_kv_type=None,
               host_bw_gbs=HOST_BW_GBS, cpu_list=a.cpu_list, threads=a.threads, loadavg_start=loadavg,
               ts=time.strftime("%FT%TZ", time.gmtime()))
    if wave_rec is not None:
        row.update(prompt_n=wave_rec["prompt_n"], cache_n=wave_rec["n_prompt_tokens_cache"], n_fork_tokens=wave_rec["n_fork_tokens"])
    else:
        row.update(prompt_n=row.get("prompt_n_by_wave"), cache_n=row.get("cache_n_by_wave"), n_fork_tokens=row.get("n_fork_tokens_by_wave"))
    row["row_sha256"] = None
    row["row_sha256"] = row_hash(row)
    return row

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
            if pmode == "poolfull_bank":
                run_poolfull_bank(a, la, ldir, corpus, Lname, srv, summary, say, facts); continue
            if pmode.startswith("bank"):
                run_bank_cell(a, la, ldir, corpus, Lname, pmode, srv, summary, say, facts, off)
                off = len(open(ldir / "server.log", "rb").read()); continue
            if la["kind"] == "poolfull":
                run_poolfull(a, la, ldir, corpus, Lname, srv, summary, say, facts); continue
            try:
                ratio = run_cell(a, la, ldir, corpus, Lname, pmode, srv, summary, say, facts, ratio, off)
            except CellRefused as e:       # fail loud: the cell sent nothing; the REFUSED_OVERSIZE row is already in summary.jsonl
                say(f"CELL_REFUSED {la['id']} {Lname} {pmode}: {e}")
                if not getattr(e, "logged", False):    # make_doc refusals do not go through guard(); record them here too
                    summary(dict(launch=la["id"], cell=f"{Lname}_{pmode}", result="REFUSED_OVERSIZE", reason=str(e)))
                a.refused = True
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
    limit = slot_limit(la, L)
    cdir = ldir / f"{Lname}_{pmode}"; cdir.mkdir(exist_ok=True)
    cl = open(cdir / "cell.log", "w"); cell_header(cl, a, la, Lname, pmode, srv, facts)
    cl.write(f"SLOT_BUDGET limit={limit} L={L} npredict={a.npredict} slot_margin={SLOT_MARGIN_TOK} suffix_reserve={DEPTH_SUFFIX_RESERVE_TOK} "
             f"template_overhead={TEMPLATE_OVERHEAD_TOK} guard_headroom={GUARD_HEADROOM_TOK} depth_doc_tok={depth_doc_tok(L, a.npredict)}\n")
    req_tok = {}          # (wave, stream) -> planned request tokens (document + suffix as tokenized); checked BEFORE any request is sent
    def guard(w, s_, n_req):
        req_tok[(w, s_)] = n_req
        if n_req + a.npredict > limit - GUARD_HEADROOM_TOK:
            msg = (f"REFUSED {la['id']} {Lname} {pmode}: wave {w} stream {s_} plans {n_req} prompt tokens + n_predict {a.npredict} = "
                   f"{n_req + a.npredict} > slot limit {limit} - guard {GUARD_HEADROOM_TOK}")
            cl.write(msg + "\n"); cl.close(); say(msg)
            summary(dict(launch=la["id"], cell=f"{Lname}_{pmode}", result="REFUSED_OVERSIZE", np=np_, L=L, slot_limit=limit,
                         planned_request_tokens=n_req, wave=w, stream=s_, n_predict=a.npredict, c=la["c"], unified=la["unified"]))
            e = CellRefused(msg); e.logged = True
            raise e
    loadavg = [round(x, 2) for x in os.getloadavg()]
    waves = []
    t_cell = time.time()
    shared_doc = None
    depth_pre = {}        # depth mode: (w, s) -> full prompt text, ALL waves built and token-counted before wave 0 is sent
    for w in range(a.waves):
        prompts = []
        if pmode in ("shared", "shared_fork"):
            if shared_doc is None:
                shared_doc, n, ratio = make_doc(corpus, a.port, ntok - 40, ratio)
                guard(0, -1, n + 40)
                warm = one_request(a.port, shared_doc + "\n\n[warm]\n", 1, "warm")
                cl.write(f"SHARED_WARM prompt_n={warm['prompt_n']} cache_n={warm['cache_n']} err={warm['error']}\n")
            for s in range(nclients):
                prompts.append(shared_doc + f"\n\n[stream {s} wave {w} nonce {hashlib.sha1(f'{la['id']}{Lname}{w}{s}'.encode()).hexdigest()[:8]}] "
                               "Summarize the document above in detail.\n")
        elif pmode == "depth":
            # prefill-once: wave 0 = distinct L-token doc per stream; waves >=1 = SAME doc + fresh short suffix (<=256 tok)
            if w == 0:
                # KVU-12.4: build + token-count EVERY wave's prompt for EVERY stream before the first request is sent; refuse (CellRefused) if any
                # would not fit the slot.  Document budget comes from the slot (depth_doc_tok), documents never exceed it (make_doc), suffixes are trimmed to the reserve.
                depth_docs, depth_n = [], []
                for s in range(nclients):
                    d, n, ratio = make_doc(corpus, a.port, depth_doc_tok(L, a.npredict), ratio)
                    depth_docs.append(d); depth_n.append(n)
                    depth_pre[(0, s)] = d + f"\n\n[stream {s} wave 0] Summarize the document above in detail.\n"
                    guard(0, s, n + n_tokens(a.port, f"\n\n[stream {s} wave 0] Summarize the document above in detail.\n") + 8)
                    for w2 in range(1, a.waves):
                        hdr = f"\n\n[stream {s} wave {w2}] Continue with these notes:\n"
                        sfx = corpus.take(int(DEPTH_SUFFIX_TOK * 0.8 * ratio))      # ~0.8*suffix tokens, fresh text per stream per wave
                        sn = n_tokens(a.port, hdr + sfx)
                        for _ in range(8):                                          # trim the suffix into its reserve
                            if sn <= DEPTH_SUFFIX_RESERVE_TOK - 8:
                                break
                            sfx = sfx[:max(1, len(sfx) - int((sn - (DEPTH_SUFFIX_RESERVE_TOK - 8)) * len(sfx) / max(sn, 1) * 1.1) - 1)]
                            sn = n_tokens(a.port, hdr + sfx)
                        depth_pre[(w2, s)] = d + hdr + sfx
                        guard(w2, s, n + sn + 8)
                cl.write(f"SLOT_FIT_OK max_planned_request_tokens={max(req_tok.values())} max_plus_npredict={max(req_tok.values()) + a.npredict} "
                         f"slot_limit={limit} min_headroom_tok={limit - max(req_tok.values()) - a.npredict} doc_tokens={depth_n}\n")
            for s in range(nclients):
                prompts.append(depth_pre[(w, s)])
        else:
            for s in range(nclients):
                d, n, ratio = make_doc(corpus, a.port, ntok - 40, ratio)
                guard(w, s, n + 40)
                prompts.append(d + f"\n\n[stream {s} wave {w} nonce {hashlib.sha1(f'{la['id']}{Lname}{w}{s}'.encode()).hexdigest()[:8]}] "
                               "Summarize the document above in detail.\n")
        bar = threading.Barrier(nclients)
        t0 = time.time()
        with ThreadPoolExecutor(nclients) as ex:
            recs = list(ex.map(lambda sp: one_request(a.port, sp[1], a.npredict, f"w{w}s{sp[0]}", bar, sp[0] if pmode == "depth" else None), enumerate(prompts)))
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
                  rss_gib=rss, vmhwm_gib=hwm, errors=[r["error"] for r in recs if r["error"]][:4],
                  n_reprefill=(sum(1 for r in recs if (r["prompt_n"] or 0) > 0.5 * ntok) if (pmode == "depth" and w > 0) else None))
        waves.append(wv)
        cl.write("WAVE " + json.dumps(wv) + "\n"); cl.flush()
        say(f"{la['id']} {Lname} {pmode} wave{w}: wall={wv['wall_s']}s ttft={wv['ttft_s']} dec_sum={wv['decode_tps_sum']} cache={wv['n_prompt_tokens_cache']} err={wv['n_err']}")
    errs, _ = scan_log(ldir / "server.log", off)
    DW = waves[1:] if pmode == "depth" else None
    def med(key, ws=None):
        v = [x for w in (waves if ws is None else ws) for x in (w[key] if isinstance(w[key], list) else [w[key]]) if x is not None]
        return round(statistics.median(v), 2) if v else None
    row = dict(launch=la["id"], cell=f"{Lname}_{pmode}", np=np_, clients=nclients, n_fork_tokens_by_wave=[w["n_fork_tokens"] for w in waves], n_reprefill_by_wave=[w["n_reprefill"] for w in waves], prompt_n_by_wave=[w["prompt_n"] for w in waves], c=la["c"], unified=la["unified"], mtp=la["mtp"], L=L, pmode=pmode,
               waves=a.waves, result="OK" if all(w["n_err"] == 0 for w in waves) else "ERRORS",
               prefill_tps_med=med("prefill_tps", waves[:1] if pmode == "depth" else None), ttft_s_med=med("ttft_s", waves[:1] if pmode == "depth" else None),
               decode_tps_stream_med=med("decode_tps_per_stream", DW), decode_tps_sum_med=med("decode_tps_sum", DW),
               decode_tps_overlap_med=med("decode_tps_overlap_window", DW), agg_e2e_med=med("agg_e2e_tok_s", DW),
               depth_waves_note=("wave 0 = prefill/TTFT (n=np streams); waves>=1 = decode at depth, NOT independent prefills" if pmode == "depth" else None),
               ttft_s_decode_waves_med=med("ttft_s", DW) if pmode == "depth" else None, prompt_n_decode_waves_med=med("prompt_n", DW) if pmode == "depth" else None, cache_n_by_wave=[w["n_prompt_tokens_cache"] for w in waves],
               rss_gib_max=max((w["rss_gib"] or 0) for w in waves), stderr_hits=errs, cell_wall_s=round(time.time() - t_cell, 1),
               physics=physics(np_, L, la["mtp"]), host_bw_gbs=HOST_BW_GBS, n_ctx_slot=facts.get("n_ctx_slot"), kv_unified=facts.get("kv_unified"),
               slot_limit=limit, max_planned_request_tokens=(max(req_tok.values()) if req_tok else None),
               min_headroom_tok=(limit - max(req_tok.values()) - a.npredict if req_tok else None))
    with open(cdir / "waves_capture.jsonl", "w") as wf:
        for w in waves:
            wf.write(json.dumps(finalize_row(dict(row, pmode=pmode, L=L, cell=row["cell"], decode_tps_sum=w["decode_tps_sum"], ttft_s=w["ttft_s"],
                                                  prefill_tps=w["prefill_tps"], decode_tps_per_stream=w["decode_tps_per_stream"],
                                                  agg_e2e_tok_s=w["agg_e2e_tok_s"]), a, la, srv, facts, loadavg, wave=w["wave"], wave_rec=w)) + "\n")
    row = finalize_row(row, a, la, srv, facts, loadavg)
    row["validation_problems"] = validate_row(row); row["row_sha256"] = row_hash(row)
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
    row = finalize_row(row, a, la, srv, facts, [round(x, 2) for x in os.getloadavg()])
    row["validation_problems"] = validate_row(row); row["row_sha256"] = row_hash(row)
    cl.write("CELL_SUMMARY " + json.dumps(row) + "\n"); cl.close()
    summary(row)


# ----------------------------------------------------------------------------------------------- CONTEXT BANK
# Operator 2026-10-20: never re-prefill large contexts per cell.  Build once: BANK_DOCS distinct documents, each grown
# additively in its own pinned slot (8k -> 32k -> 64k -> 128k, token-ID prompts so prompt_n == chunk length), saving
# the slot after each step (/slots/{id}?action=save).  Every cell then restores the needed doc{i}_L{name}.bin and decodes.
DEFAULT_BANK_DIR = str(ART / "_bank_scratch")        # NOT in git (.gitignore '*'); manifest IS committed
MANIFEST = ART / "bank_manifest.json"
def bank_sub(bank_dir, unified, np_):
    """Slot-state files embed n_stream (restore fails with 'n_stream mismatch' across KV modes AND across split np), so banks are per layout:
    unified (n_stream 1: valid for every np and pool) and split_np{N}."""
    return str(Path(bank_dir) / ("unified" if unified else f"split_np{np_}"))
def manifest_path(a, unified, np_):
    base = Path(a.manifest)
    return base.with_name(base.stem + ("_unified" if unified else f"_split_np{np_}") + base.suffix)

def slot_action(port, slot, action, filename=None):
    body = {"filename": filename} if filename else {}
    return http(port, f"/slots/{slot}?action={action}", body, timeout=3600)

def tokenize_text(port, text):
    return http(port, "/tokenize", {"content": text, "add_special": False})["tokens"]

def make_bank_tokens(port, corpus, ndocs, maxtok, bankdir):
    docs = []
    for i in range(ndocs):
        f = Path(bankdir) / f"doc{i}.tokens.json"
        if f.exists():
            docs.append(json.loads(f.read_text())); continue
        nch = int(maxtok * 4.2); toks = []
        while len(toks) < maxtok:
            toks += tokenize_text(port, corpus.take(nch)); nch = max(nch // 2, 20000)
        toks = toks[:maxtok]
        f.write_text(json.dumps(toks)); docs.append(toks)
    return docs

def suffix_tokens(port, corpus, n=200):
    t = tokenize_text(port, corpus.take(n * 6))
    return t[:n]

@__import__("contextlib").contextmanager
def server_ctx(a, la, ldir, bankdir):
    """Launch one production-argv server (+ --slot-save-path bankdir) as a child; yield (srv, facts, say); always stop by PID."""
    ldir.mkdir(parents=True, exist_ok=True)
    argv = prod_argv(la["np"], la["c"], la["unified"], la["mtp"], a.port, a.threads, la.get("bin") or a.binary,
                     tuple(la.get("extra", ())) + ("--slot-save-path", str(bankdir)))
    env = dict(os.environ); env.update(PROD_ENV)
    prefix = ["numactl", "--interleave=all", "--", "taskset", "-c", a.cpu_list]
    log = open(ldir / "launch.log", "a")
    def say(x):
        line = f"[{time.strftime('%FT%TZ', time.gmtime())}] {x}"; print(line, flush=True); log.write(line + "\n"); log.flush()
    say(f"LAUNCH {la['id']} argv={json.dumps(prefix + argv)}")
    srv = Server(argv, env, prefix, ldir, a.port); srv.bin_dir = os.path.dirname(os.path.realpath(argv[0])); srv.start()
    try:
        st = srv.healthy(); say(f"health={st}")
        if st != "HEALTHY":
            raise RuntimeError(f"server {st}")
        link = srv.linkage(); say(f"LINKAGE {json.dumps(link)}")
        if not link.get("all_in_kernel_store"):
            raise RuntimeError("linkage mismatch")
        slog = (ldir / "server.log").read_text(errors="ignore")
        facts = {k: (re.findall(pt, slog) or [None])[-1] for k, pt in LOAD_PATS.items()}
        facts["rss_after_load_gib"] = srv.rss_gib()[0]; say(f"LOADFACTS {json.dumps(facts)}")
        yield srv, facts, say
    finally:
        dead = srv.stop(); say(f"STOP dead={dead} pid={srv.proc.pid}")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()

def build_bank(a, outroot, corpus):
    """--build-bank [--bank-kv unified|split --bank-np N]: grow documents additively, save after each step, np1 controls, manifest."""
    uni = (a.bank_kv == "unified"); root = Path(a.bank_dir); root.mkdir(parents=True, exist_ok=True)
    names = a.bank_names.split(","); nd = a.bank_docs if uni else a.bank_np
    sub = Path(bank_sub(a.bank_dir, uni, nd)); sub.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("*\n")
    maxD = bank_tok(names[-1])
    c = (nd * (maxD + 1024) + 131072) if uni else nd * LS[names[-1]]
    la = dict(id=("bank_build_unified_np%d_pool%d" % (nd, c)) if uni else ("bank_build_split_np%d_c%d" % (nd, c)), np=nd, c=c, unified=uni, mtp=True, cells=[], kind="bank_build",
              extra=("--no-cache-idle-slots",) if uni else ())
    steps, manifest = [], {"schema": "epyc.np_ctx_kvu_cpu.bank_manifest.v1", "kv": a.bank_kv, "np": nd, "names": names, "ndocs": nd, "docs": {}, "controls": [],
                           "binary_sha256": file_sha256(os.path.realpath(a.binary)), "model": MODEL, "built": time.strftime("%FT%TZ", time.gmtime())}
    with server_ctx(a, la, outroot / la["id"], sub) as (srv, facts, say):
        manifest["n_ctx_slot"] = facts.get("n_ctx_slot"); manifest["kv_unified_from_log"] = facts.get("kv_unified")
        docs = make_bank_tokens(a.port, corpus, nd, maxD, root)
        for i, toks in enumerate(docs):
            manifest["docs"][str(i)] = dict(tokens_sha256=hashlib.sha256(json.dumps(toks).encode()).hexdigest(), files={})
            prev = 0
            for nm in names:
                D = bank_tok(nm)
                r = one_request(a.port, toks[:D], a.bank_npredict, f"bank_d{i}_{nm}", id_slot=i, stream=False, timeout=14400)
                fn = f"doc{i}_L{nm}.bin"
                sv = slot_action(a.port, i, "save", fn)
                step = dict(doc=i, name=nm, depth_from=prev, depth_to=D, chunk=D - prev, prompt_n=r["prompt_n"], cache_n=r["cache_n"],
                            prefill_ms=r["prompt_ms"], prefill_tps=r["prompt_tps"], wall_s=round(r["t_end"] - r["t_start"], 1),
                            saved_tokens=sv.get("n_saved", sv.get("n_tokens")), save_bytes=sv.get("n_written", sv.get("n_bytes")),
                            save_ms=(sv.get("timings") or {}).get("save_ms"), error=r["error"], file=fn)
                steps.append(step); say("BANKSTEP " + json.dumps(step))
                manifest["docs"][str(i)]["files"][nm] = dict(file=fn, n_tokens=D, step=step)
                prev = D
        for nm in names:             # np1 one-shot controls (single active stream on slot 0)
            D = bank_tok(nm)
            slot_action(a.port, 0, "erase")
            r = one_request(a.port, docs[0][:D], a.bank_npredict, f"ctl_{nm}", id_slot=0, stream=False, timeout=14400)
            chunk_ms = sum(st["prefill_ms"] or 0 for st in steps if st["doc"] == 0 and bank_tok(st["name"]) <= D)
            ctl = dict(name=nm, D=D, oneshot_ms=r["prompt_ms"], oneshot_prompt_n=r["prompt_n"], chunk_sum_ms=chunk_ms,
                       ratio_chunksum_over_oneshot=(round(chunk_ms / r["prompt_ms"], 3) if r["prompt_ms"] else None))
            manifest["controls"].append(ctl); say("CONTROL " + json.dumps(ctl))
    for i in manifest["docs"]:
        for nm, e in manifest["docs"][i]["files"].items():
            fp = sub / e["file"]
            if fp.exists():
                e["sha256"] = sha256_file(fp); e["bytes"] = fp.stat().st_size
    manifest["total_prefill_s"] = round(sum((st["prefill_ms"] or 0) for st in steps) / 1000, 1)
    mp = manifest_path(a, uni, nd); mp.write_text(json.dumps(manifest, indent=1))
    print(f"BANK DONE manifest={mp} total_prefill_s={manifest['total_prefill_s']}", flush=True)

def load_bank_tokens(bankdir, i):
    return json.loads((Path(bankdir) / f"doc{i}.tokens.json").read_text())

def parse_bank_mode(pmode, np_):
    """'bank' | 'bank<K>' (restore K, decode K) | 'bank_resident<K>' (restore np, decode K)."""
    m = re.fullmatch(r"bank(_resident)?(\d*)", pmode)
    active = int(m.group(2)) if m.group(2) else np_
    return active, (np_ if m.group(1) else active)          # (active streams, restored slots)

def run_bank_cell(a, la, ldir, corpus, Lname, pmode, srv, summary, say, facts, off):
    L, np_ = LS[Lname], la["np"]; D = bank_tok(Lname)
    active, nrest = parse_bank_mode(pmode, np_)
    cdir = ldir / f"{Lname}_{pmode}"; cdir.mkdir(exist_ok=True)
    cl = open(cdir / "cell.log", "w"); cell_header(cl, a, la, Lname, pmode, srv, facts)
    loadavg = [round(x, 2) for x in os.getloadavg()]
    docs = [load_bank_tokens(a.bank_dir, i) for i in range(nrest)]
    waves, t_cell = [], time.time()
    for w in range(a.waves):
        rest = []
        for i in range(nrest):                                    # restore BEFORE every wave (untimed): waves are identical, no drift in reuse
            fn = f"doc{i}_L{Lname}.bin"
            t0 = time.time(); r = slot_action(a.port, i, "restore", fn)
            rest.append(dict(slot=i, n_restored=r.get("n_restored", r.get("n_tokens")), ms=round((time.time() - t0) * 1000)))
        sfx = [suffix_tokens(a.port, corpus, 200) for _ in range(active)]
        bar = threading.Barrier(active); t0 = time.time()
        with ThreadPoolExecutor(active) as ex:
            recs = list(ex.map(lambda sp: one_request(a.port, docs[sp[0]][:D] + sp[1], a.npredict, f"w{w}s{sp[0]}", bar, sp[0]), enumerate(sfx)))
        wall = time.time() - t0; rss, hwm = srv.rss_gib()
        ok = [r for r in recs if r["error"] is None and r["status"] == 200]; dec = [r["decode_tps"] for r in ok if r["decode_tps"]]
        resident = sum(x["n_restored"] or 0 for x in rest)
        wv = dict(wave=w, wall_s=round(wall, 1), n_ok=len(ok), n_err=len(recs) - len(ok), active_streams=active, restored_slots=nrest,
                  resident_tokens_at_start=resident, used_cells_est=resident + active * (200 + a.npredict), restore_ms=[x["ms"] for x in rest],
                  ttft_s=[round(r["ttft"], 2) if r["ttft"] else None for r in recs], prefill_tps=[round(r["prompt_tps"], 1) if r["prompt_tps"] else None for r in recs],
                  prompt_n=[r["prompt_n"] for r in recs], n_fork_tokens=[r["n_fork_tokens"] for r in recs], n_prompt_tokens_cache=[r["cache_n"] for r in recs],
                  predicted_n=[r["predicted_n"] for r in recs], decode_tps_per_stream=[round(x, 2) for x in dec], decode_tps_sum=round(sum(dec), 2),
                  decode_tps_overlap_window=(round(overlap_agg(recs), 2) if overlap_agg(recs) else None),
                  agg_e2e_tok_s=round(sum(r["predicted_n"] or 0 for r in ok) / wall, 2),
                  draft_n=[r["draft_n"] for r in recs], draft_n_accepted=[r["draft_n_accepted"] for r in recs],
                  rss_gib=rss, vmhwm_gib=hwm, errors=[r["error"] for r in recs if r["error"]][:4],
                  n_reprefill=sum(1 for r in recs if (r["prompt_n"] or 0) > 0.5 * D))
        waves.append(wv); cl.write("WAVE " + json.dumps(wv) + "\n"); cl.flush()
        say(f"{la['id']} {Lname} {pmode} wave{w}: wall={wv['wall_s']}s ttft={wv['ttft_s']} dec_sum={wv['decode_tps_sum']} prompt_n={wv['prompt_n']} resident={resident} err={wv['n_err']}")
    errs, _ = scan_log(ldir / "server.log", off)
    def med(key):
        v = [x for w in waves for x in (w[key] if isinstance(w[key], list) else [w[key]]) if x is not None]
        return round(statistics.median(v), 2) if v else None
    row = dict(launch=la["id"], cell=f"{Lname}_{pmode}", np=np_, clients=active, active_streams=active, restored_slots=nrest, c=la["c"],
               unified=la["unified"], mtp=la["mtp"], L=L, pmode=pmode, waves=a.waves, result="OK" if all(w["n_err"] == 0 for w in waves) else "ERRORS",
               source="bank", bank_manifest_sha256=(sha256_file(manifest_path(a, la["unified"], la["np"])) if manifest_path(a, la["unified"], la["np"]).exists() else None),
               ttft_s_med=med("ttft_s"), decode_tps_stream_med=med("decode_tps_per_stream"), decode_tps_sum_med=med("decode_tps_sum"),
               decode_tps_overlap_med=med("decode_tps_overlap_window"), agg_e2e_med=med("agg_e2e_tok_s"),
               resident_tokens_by_wave=[w["resident_tokens_at_start"] for w in waves], used_cells_est_by_wave=[w["used_cells_est"] for w in waves],
               prompt_n_by_wave=[w["prompt_n"] for w in waves], cache_n_by_wave=[w["n_prompt_tokens_cache"] for w in waves],
               n_fork_tokens_by_wave=[w["n_fork_tokens"] for w in waves], n_reprefill_by_wave=[w["n_reprefill"] for w in waves],
               draft_n_by_wave=[w["draft_n"] for w in waves], draft_n_accepted_by_wave=[w["draft_n_accepted"] for w in waves],
               rss_gib_max=max((w["rss_gib"] or 0) for w in waves), stderr_hits=errs, cell_wall_s=round(time.time() - t_cell, 1),
               physics=physics(active, L, la["mtp"]), host_bw_gbs=HOST_BW_GBS, kv_unified=facts.get("kv_unified"))
    with open(cdir / "waves_capture.jsonl", "w") as wf:
        for w in waves:
            wf.write(json.dumps(finalize_row(dict(row, decode_tps_sum=w["decode_tps_sum"], ttft_s=w["ttft_s"]), a, la, srv, facts, loadavg, wave=w["wave"], wave_rec=w)) + "\n")
    row = finalize_row(row, a, la, srv, facts, loadavg); row["validation_problems"] = validate_row(row); row["row_sha256"] = row_hash(row)
    cl.write("CELL_SUMMARY " + json.dumps(row) + "\n"); cl.close(); summary(row)

def run_poolfull_bank(a, la, ldir, corpus, Lname, srv, summary, say, facts):
    """Pool-full stress without re-prefill of the resident part: restore pool//D docs (fills the pool), then send np streams:
    resident ones decode; the rest bring fresh bank-token prompts of D tokens that cannot fit."""
    D = bank_tok(Lname); np_ = la["np"]; nfit = max(1, la["c"] // LS[Lname])
    cdir = ldir / f"{Lname}_poolfull_bank"; cdir.mkdir(exist_ok=True)
    cl = open(cdir / "cell.log", "w"); cell_header(cl, a, la, Lname, "poolfull_bank", srv, facts)
    cl.write(f"POOLFULL demand={np_ * LS[Lname]} pool={la['c']} oversub={np_ * LS[Lname] / la['c']:.2f}x resident_docs={nfit}\n")
    rest = [slot_action(a.port, i, "restore", f"doc{i}_L{Lname}.bin").get("n_restored") for i in range(nfit)]
    docs = [load_bank_tokens(a.bank_dir, i) for i in range(np_)]
    sfx = [suffix_tokens(a.port, corpus, 200) for _ in range(np_)]
    stop_poll = threading.Event(); polls = []
    t0 = time.time()
    def poll():
        while not stop_poll.is_set():
            try:
                sl = http(a.port, "/slots", timeout=5)
                polls.append(dict(t=round(time.time() - t0, 1), slots=[dict(id=x.get("id"), proc=x.get("is_processing"), n_ctx=x.get("n_ctx")) for x in sl]))
            except Exception as e:
                polls.append(dict(t=round(time.time() - t0, 1), err=repr(e)[:80]))
            stop_poll.wait(15)
    th = threading.Thread(target=poll, daemon=True); th.start()
    off = len(open(ldir / "server.log", "rb").read()); bar = threading.Barrier(np_)
    with ThreadPoolExecutor(np_) as ex:
        recs = list(ex.map(lambda sp: one_request(a.port, docs[sp[0]][:D] + sp[1], a.npredict, f"pf{sp[0]}", bar, sp[0], True, a.stress_timeout), enumerate(sfx)))
    stop_poll.set(); th.join(10); errs, _ = scan_log(ldir / "server.log", off)
    per = [dict(tag=r["tag"], resident=(i < nfit), status=r["status"], error=r["error"], ttft=r["ttft"], prompt_n=r["prompt_n"], cache_n=r["cache_n"],
                predicted_n=r["predicted_n"], truncated=r["truncated"], stop_type=r["stop_type"], wall=round(r["t_end"] - r["t_start"], 1)) for i, r in enumerate(recs)]
    outcome = ("ERROR" if any(x["status"] != 200 or x["error"] for x in per if not x["resident"]) else
               "TRUNCATE" if any(x["truncated"] for x in per) else "QUEUE_OR_FIT")
    row = dict(launch=la["id"], cell=f"{Lname}_poolfull_bank", np=np_, c=la["c"], L=LS[Lname], pmode="poolfull_bank", result="POOLFULL", outcome=outcome,
               restored_tokens=rest, per_request=per, stderr_hits=errs, slots_polls=polls[:200], rss_gib_max=srv.rss_gib()[1])
    row = finalize_row(row, a, la, srv, facts, [round(x, 2) for x in os.getloadavg()])
    row["validation_problems"] = validate_row(row); row["row_sha256"] = row_hash(row)
    cl.write("CELL_SUMMARY " + json.dumps(row) + "\n"); cl.close(); summary(row)

def run_gates(a, outroot, corpus):
    """Correctness gates for the bank (small L): (a) restore+suffix output byte-identical to live prefill+suffix,
    (b) MTP draft acceptance restored vs live, (c) restored slot extended by a chunk gives prompt_n == chunk."""
    bankdir = Path(a.bank_dir); bankdir.mkdir(parents=True, exist_ok=True); (bankdir / ".gitignore").write_text("*\n")
    D, CH, NP = a.gate_tokens, a.gate_chunk, a.gate_npredict
    results = []
    for mode in ("split", "unified"):
        c = 2 * (D + 4096)
        la = dict(id=f"gate_{mode}_np2", np=2, c=c, unified=(mode == "unified"), mtp=True, cells=[], kind="gate",
                  extra=(("--no-cache-idle-slots",) if mode == "unified" else ()))
        with server_ctx(a, la, outroot / la["id"], bankdir) as (srv, facts, say):
            toks = make_bank_tokens(a.port, corpus, 1, D + CH, bankdir)[0]
            base = toks[:D]; chunk = toks[:D + CH]
            sfxA, sfxB = suffix_tokens(a.port, corpus, 128), suffix_tokens(a.port, corpus, 128)
            g = dict(mode=mode, D=D, n_ctx_slot=facts.get("n_ctx_slot"), kv_unified=facts.get("kv_unified"))
            def req(tokens, npred, tag, stream=False):
                r = one_request(a.port, tokens, npred, tag, id_slot=0, stream=stream, timeout=3600)
                return r
            for sfx_name, sfx in (("A", sfxA), ("B", sfxB)):
                slot_action(a.port, 0, "erase")
                live = req(base + sfx, NP, f"live{sfx_name}")
                slot_action(a.port, 0, "erase")
                pre = req(base, a.bank_npredict, "prefill_only")
                sv = slot_action(a.port, 0, "save", "gate_base.bin")
                slot_action(a.port, 0, "erase")
                rs = slot_action(a.port, 0, "restore", "gate_base.bin")
                rest = req(base + sfx, NP, f"restored{sfx_name}")
                g[f"live{sfx_name}"] = dict(prompt_n=live["prompt_n"], draft_n=live["draft_n"], draft_acc=live["draft_n_accepted"], predicted_n=live["predicted_n"], decode_tps=live["decode_tps"], err=live["error"])
                g[f"restored{sfx_name}"] = dict(prompt_n=rest["prompt_n"], cache_n=rest["cache_n"], draft_n=rest["draft_n"], draft_acc=rest["draft_n_accepted"], predicted_n=rest["predicted_n"], decode_tps=rest["decode_tps"], err=rest["error"])
                g[f"identical{sfx_name}"] = (live["text"] == rest["text"])
                g[f"text_prefix_match_chars{sfx_name}"] = next((i for i, (x, y) in enumerate(zip(live["text"], rest["text"])) if x != y), min(len(live["text"]), len(rest["text"])))
                g["save"] = dict(n_tokens=sv.get("n_saved", sv.get("n_tokens")), bytes=sv.get("n_written", sv.get("n_bytes")), raw=sv)
                g["restore"] = dict(raw=rs)
                g["prefill_only"] = dict(prompt_n=pre["prompt_n"], err=pre["error"], status=pre["status"])
            # (c) restored base extended by a CH-token chunk
            slot_action(a.port, 0, "erase"); slot_action(a.port, 0, "restore", "gate_base.bin")
            ext = req(chunk, 1, "extend")
            g["extend"] = dict(prompt_n=ext["prompt_n"], cache_n=ext["cache_n"], expected_prompt_n=CH, ok=(ext["prompt_n"] == CH), err=ext["error"])
            results.append(g); say("GATE " + json.dumps(g))
    (outroot / "gates.json").write_text(json.dumps(results, indent=1)); print("GATES DONE", json.dumps(results)[:400], flush=True)

# ----------------------------------------------------------------------------------------------- main
def main():
    global MANIFEST
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print every launch+cell argv and estimates; launch nothing")
    p.add_argument("--verify-against-launcher", action="store_true")
    p.add_argument("--out", default=str(ART / "runs"))
    p.add_argument("--kv-modes", default="split,unified")
    p.add_argument("--nps", default="1,2,4,8")
    p.add_argument("--lens", default="8k,32k,64k,128k")
    p.add_argument("--pmodes", default="independent,shared", help="add shared_fork (needs --fork-binary at run time; dry-run prints a placeholder) for the prefix-fork arm")
    p.add_argument("--pools", default="262144,524288,1048576")
    p.add_argument("--waves", type=int, default=None, help="default 3; 4 under --tier0 (1 prefill wave + 3 decode-at-depth waves)")
    p.add_argument("--build-bank", action="store_true", help="build the context bank once (prefill 8 docs additively; save per step; np1 controls; manifest)")
    p.add_argument("--from-bank", action="store_true", help="tier0: bank cells restore doc{i}_L{L}.bin instead of prefilling (default when --tier0)")
    p.add_argument("--gates", action="store_true", help="bank correctness gates at small L (restore==live output, MTP acceptance, chunk extend)")
    p.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    p.add_argument("--manifest", default=str(MANIFEST), help="bank manifest path (committed; bank files themselves are not)")
    p.add_argument("--bank-names", default=",".join(BANK_NAMES))
    p.add_argument("--bank-docs", type=int, default=BANK_DOCS)
    p.add_argument("--bank-kv", default="unified", choices=["unified", "split"], help="layout of the bank to build; files are per layout (n_stream must match on restore)")
    p.add_argument("--bank-np", type=int, default=4, help="split bank: np (= number of docs) of the layout")
    p.add_argument("--bank-npredict", type=int, default=0, help="n_predict for bank prefill steps (0 = prefill only)")
    p.add_argument("--gate-tokens", type=int, default=7680); p.add_argument("--gate-chunk", type=int, default=1024); p.add_argument("--gate-npredict", type=int, default=256)
    p.add_argument("--stress-timeout", type=int, default=1800)
    p.add_argument("--tier0", action="store_true", help="decision-core cell list (fixed order, ABA), prefill-once depth mode")
    p.add_argument("--npredict", type=int, default=N_PREDICT_DEFAULT)
    p.add_argument("--mtp-off", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--anchor", action="store_true", help="add the T=8-clients/np4 short-ctx cell comparable to stack_topology.yaml:140-150")
    p.add_argument("--binary", default=str(BIN), help="server binary for the plain arms (default: production v10 kernel store)")
    p.add_argument("--extra-argv", default="", help="extra server argv for ALL arms (space separated)")
    p.add_argument("--fork-binary", default="", help="cross-slot prefix-fork build; enables the shared_fork prompt-mode arm")
    p.add_argument("--fork-min-tokens", type=int, default=1024, help="value for --slot-fork-min-tokens in the shared_fork arm (0 = off)")
    p.add_argument("--aba-key", default="", help="KVU-12.4: A/B pairing key; with --aba-ledger, refuse a launch whose argv differs (beyond KV mode) from an earlier launch with the same key")
    p.add_argument("--aba-ledger", default="", help="JSONL of normalized argv per key (shared across driver invocations)")
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
    MANIFEST = Path(a.manifest)
    if a.waves is None:
        a.waves = 4 if a.tier0 else 3
    ls = build_tier0(a) if a.tier0 else order_launches(build_launches(a), a)
    if a.from_bank and not a.tier0:          # full matrix on the bank: INDEPENDENT/depth cells restore instead of prefilling
        for la in ls:
            la["cells"] = [(l, ("bank" if (pm in ("independent", "depth") and l != "t8short") else pm)) for l, pm in la["cells"]]
            if any(pm.startswith("bank") for _, pm in la["cells"]):
                la["extra"] = tuple(la.get("extra", ())) + ("--slot-save-path", bank_sub(a.bank_dir, la["unified"], la["np"]))
    if a.only:
        keys = a.only.split(","); ls = [l for l in ls if any(k in l["id"] for k in keys)]
    aba_guard(a, ls); static_fit_check(a, ls)
    if a.dry_run:
        tot, ncell = 0.0, 0
        for la in ls:
            argv = prod_argv(la["np"], la["c"], la["unified"], la["mtp"], a.port, a.threads, la.get("bin") or a.binary, la.get("extra", ()))
            print(f"\n== launch {la['id']} kind={la['kind']} np={la['np']} c={la['c']} unified={la['unified']} mtp={la['mtp']}")
            print("   ARGV " + " ".join(["numactl", "--interleave=all", "--", "taskset", "-c", a.cpu_list] + argv))
            print("   ENV  " + " ".join(f"{k}={v}" for k, v in PROD_ENV.items()))
            tot += 120
            for (l, pm) in la["cells"]:
                if pm.startswith("bank"):
                    e = est_bank_cell(la["np"], l, pm, a.waves, a.npredict, la["mtp"])
                elif pm == "poolfull_bank":
                    e = 4 * (LS[l] * KV_BYTES_PER_TOKEN / 1.5e9) + 4 * (LS[l] / prefill_tps(LS[l])) * 0.25 + a.stress_timeout * 0.5
                elif la["kind"] != "poolfull":
                    e = est_cell_s(la["np"], l, pm, a.waves, a.npredict, la["mtp"])
                else:
                    e = la["np"] * (LS[l] / prefill_tps(LS[l]))
                tot += e; ncell += 1
                print(f"   cell {l:>4} {pm:<11} est {e/60:7.1f} min  physics {physics(la['np'], LS[l], la['mtp'])}")
        print(f"\nTOTAL launches={len(ls)} cells={ncell} est={tot/3600:.1f} h (cells only; estimator is an ASSUMPTION model; smoke calibrates prefill_tps)")
        if a.tier0 or a.build_bank:
            sp_nps = sorted({la["np"] for la in ls if not la["unified"] and any(pm.startswith("bank") for _, pm in la["cells"])})
            tb = est_bank_build(a.bank_docs) if any(la["unified"] for la in ls) else 0.0
            print(f"BANK BUILD unified (once): est={tb/3600:.1f} h  (8 docs x 8k/32k/64k/128k additive + np1 controls)")
            tsp = 0.0
            for n_ in sp_nps:
                Ls_ = [pm_l for la in ls if not la["unified"] and la["np"] == n_ for pm_l, pm in la["cells"] if pm.startswith("bank")]
                nm_ = [nm for nm in BANK_NAMES if LS[nm] <= max(LS[x] for x in Ls_)]
                e_ = est_bank_build(n_, nm_); tsp += e_
                print(f"BANK BUILD split np{n_} (layout-specific, n_stream={n_}): est={e_/3600:.1f} h  names={nm_}  cmd: --build-bank --bank-kv split --bank-np {n_} --bank-names {','.join(nm_)}")
            print(f"BANK BUILD TOTAL est={(tb+tsp)/3600:.1f} h; bank dir {a.bank_dir}")
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(a.corpus_roots.split(","))
    sumf = out / "summary.jsonl"
    def summary(row):
        row.setdefault("ts", time.strftime("%FT%TZ", time.gmtime()))
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
    if a.build_bank:
        build_bank(a, out, corpus); print("DONE", flush=True); return
    if a.gates:
        run_gates(a, out, corpus); print("DONE", flush=True); return
    a.refused = False
    for la in ls:
        run_launch(a, la, out, corpus, summary)
    print("DONE", flush=True)
    if a.refused:
        sys.exit(3)          # a REFUSED_OVERSIZE cell is a failed cell, never a quiet pass

if __name__ == "__main__":
    main()
