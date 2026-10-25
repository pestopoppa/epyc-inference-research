#!/usr/bin/env python3
"""v11_driver_086v2.py -- COPY of v11_driver.py; ONLY item 086 differs (V11-086 v2, un-held 2026-10-22, AGENT_ID=v11-086-unhold).
086 v2: arm build = kernels/builds/cpu-20261022-8110933ba-fa-vkq (experimental/fa-vkq-reg-on-c-20261022, C + one port commit of
5bde1b83d+13bdbdc4d); workload RETARGETED from DS41 to Qwen3.8-Flash-Next (architect_general, f16 KV, D=256): llama-bench
pp5/tg32 @ d0/d32768 x arms C / B-unset / B-REG=auto (3 rotated launches each) + the q38fn serving gate (REG=auto, 5 pairs).
Correctness (tbo FA, FA harness, bit-identity, finite) runs in the BUILD filler (jobs/v11_086_fa_vkq_build_job.sh) and is read
here from INTEGRITY.json.  Rule: predeclared/086v2/PREDECLARED_086v2.md (sha-pinned as 086v2/PREDECLARED_086v2.md).
Items 052/050/051 are byte-for-byte the original code paths; use v11_driver.py for them.

v11_driver.py -- measurement drivers for V11-052 / V11-050 / V11-051 (GPU) and V11-086 (CPU).

    v11_driver.py ITEM --dry-run   [--run-dir D]   resolve builds/recipes, validate argv, print the plan; launches NOTHING
    v11_driver.py ITEM --execute   [--run-dir D]   run the measurement (REFUSES outside a filler/region-lock claim)
    v11_driver.py ITEM --decide RESULTS.json       apply the predeclared rule to a results file (also used by --selftest)
    v11_driver.py --selftest                       synthetic-data tests of every decision rule + parsers (no builds, no host)
    v11_driver.py --pin                            (re)write predeclared/PINNED.sha256 from the PREDECLARED.md files

ITEM is 052 | 050 | 051 | 086.  Exit codes: 0 ok, 3 refused (not inside a claim / preflight / PREDECLARED mismatch),
5 build-missing (arm build not in kernels/builds yet), 6 port/stack conflict, 7 proof failed (linkage/residency/port ownership),
64 usage.  The decision rule of every item lives in predeclared/<item>/PREDECLARED.md and in the decide_* function below; the
PREDECLARED.md sha256 is pinned in predeclared/PINNED.sha256 and checked before any run.  A run writes
  <run-dir>/{PREDECLARED.md, manifest.json, results.json, DECISION.json, logs...}  and prints a final  decision=<value>  line.

No pkill/pgrep: every process is our own child (Popen), TERM -> KILL -> verified dead.  Never touches kernels/production or any
production tree; arm builds are read from /mnt/raid0/llm/kernels/builds.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, shlex, signal, statistics, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir from sys.path
from fa086v2_integrity import ANSI_RE as _ANSI_RE, parse_tbo_text  # noqa: E402
from bench_protocol_086 import BenchProtocolError, validate_bench_protocol  # noqa: E402

HERE = Path(__file__).resolve().parent
K = Path("/mnt/raid0/llm/kernels/builds")
C_GPU = K / "gpu-20261020-f0c96dc1f"          # candidate C (the champion the arms are judged against)
C_CPU = K / "cpu-20261020-f0c96dc1f"
VERIFY = Path("/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh")
REC8083 = Path("/mnt/raid0/llm/epyc-orchestrator/logs/server_launches/8083.json")
MODELS = Path("/mnt/raid0/llm/models")
M27_Q8 = MODELS / "Qwen3.8-27B-Q8_0.gguf"
M27_Q4 = MODELS / "Qwen3.8-27B-UD-Q4_K_M.gguf"
GAB_CANDIDATES = [Path("/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/gpu_champion_ab.sh"),
                  Path("/workspace/repos/epyc-inference-research/scripts/benchmark/gpu_champion_ab.sh"),
                  Path("/mnt/raid0/llm/worktrees/research-ab-probe-restore/scripts/benchmark/gpu_champion_ab.sh")]
GATE_COMPARE_DIR = Path("/mnt/raid0/llm/tmp/fold-allkeeps-20261004")
FA_H1 = Path("/mnt/raid0/llm/tmp/fa-maskskip-20261003/harness")
FA_H2 = Path("/mnt/raid0/llm/tmp/fa-maskskip-batched-20261004/harness")
RUNS = Path("/mnt/raid0/llm/tmp/v11-runs")
GPU_THREADS = "184-191"                        # host threads of the GPU runner (production 8083 argv: taskset 184-191, membind 3)
LD_GPU_TAIL = ["/opt/rocm/lib", "/usr/lib/llvm-20/lib", "/opt/AMD/aocc-compiler-5.0.0/lib"]

# ledger floors (predeclared constants; see PREDECLARED.md)
TG_FLOOR, PP_FLOOR, DS41_FLOOR = 2.00, 9.91, 4.533
LABELS = {"052": ("kqroute", "gpu"), "050": ("fp32vkq", "gpu"), "051": ("d256cap", "gpu"), "086": ("fa-vkq", "cpu")}
PRE_DIR = HERE / "predeclared"
# ---- 086 v2 constants
B086 = K / "cpu-20261022-8110933ba-fa-vkq"
CORR086 = Path("/mnt/raid0/llm/tmp/v11-runs/086v2/correctness")
PRE086 = ("086v2", "PREDECLARED_086v2.md")
Q38_TRUNK = MODELS / "unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
Q38_PREFIX = ["taskset", "-c", "0-95", "numactl", "--interleave=all", "--"]   # qwen38_flash_next_recipe.SERVE_PREFIX
Q38_GGML_ENV = {"GGML_IQK": "1", "GGML_FUSED_DECODE_OFF": "1", "GGML_FA_SPLIT_KV": "0", "GGML_NOHUGEPAGE_PROCESS": "1"}  # CHAMPION_GGML_ENV
Q38_BENCH_ARGS = ["-t", "48", "-fa", "1", "-ctk", "f16", "-ctv", "f16", "-mmp", "0", "-p", "5", "-n", "32", "-d", "0,32768", "-r", "3", "-o", "json"]
CELLS086 = ("pp5", "tg32", "pp5@d32768", "tg32@d32768")
LONG086 = ("pp5@d32768", "tg32@d32768")
MIN_FLOOR086 = 2.00          # predeclared minimum bar (percent of median tok/s); NOT a calibrated CPU floor
ROT086 = (("C", "Bunset", "Bauto"), ("Bunset", "Bauto", "C"), ("Bauto", "C", "Bunset"))

class Refuse(Exception):
    def __init__(self, rc, msg): super().__init__(msg); self.rc = rc

def die(rc, msg): raise Refuse(rc, msg)
def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def say(msg, log=None):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    print(line, flush=True)
    if log: open(log, "a").write(line + "\n")
def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ----------------------------------------------------------------------------------------------- builds
STANDIN = False     # dry-run only (V11_DRY_ARM_STANDIN=1): use champion C as the arm build to exercise argv/recipe paths while the real arm is unbuilt

def arm_build(item: str) -> Path:
    """Store dir of the item's arm build. build_arm clones to llama-measure-<label>-20261020 and names the store dir by that HEAD."""
    label, dev = LABELS[item]
    if item == "086":   # v2: fixed store dir of the re-port build (job v11_086_fa_vkq_build_job.sh)
        if not (B086 / "SHA256SUMS").exists() or not (B086 / "bin" / "llama-bench").exists():
            die(5, f"build-missing: {B086} (v11-086-fa-vkq-build has not finished)")
        return B086
    src = Path(f"/mnt/raid0/llm/worktrees/llama-measure-{label}-20261020")
    if STANDIN and not (src / ".git").exists(): return require_c(dev)
    if not (src / ".git").exists():
        die(5, f"build-missing: arm source clone {src} absent (v11-arm-builds has not reached {label})")
    sha = subprocess.check_output(["git", "-C", str(src), "rev-parse", "--short=9", "HEAD"], text=True).strip()
    d = K / f"{dev}-20261020-{sha}"
    if STANDIN and not (d / "SHA256SUMS").exists(): return require_c(dev)
    if not (d / "SHA256SUMS").exists() or not (d / "bin" / "llama-server").exists():
        die(5, f"build-missing: {d} (no SHA256SUMS/bin/llama-server; arm {label} still building)")
    return d

def require_c(dev: str) -> Path:
    c = C_GPU if dev == "gpu" else C_CPU
    if not (c / "bin" / "llama-server").exists(): die(5, f"build-missing: champion {c}")
    return c

def build_sha(d: Path) -> str: return d.name.split("-")[-1]

def lib_env(build: Path, dev: str, extra: dict | None = None) -> dict:
    ld = [str(build / "bin")] + (LD_GPU_TAIL if dev == "gpu" else [])
    e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"), "LD_LIBRARY_PATH": ":".join(ld)}
    if dev == "cpu":
        # CANONICAL_OMP_ENV (epyc-inference-research scripts/lib/canonical_recipe.py) -- OMP env stack is not optional on the CPU surface
        e.update({"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "active", "OMP_DYNAMIC": "false", "GGML_IQK": "1"})
    e.update(extra or {})
    return e

def linkage_proof(build: Path, exe: str = "llama-server") -> dict:
    """verify_ggml_linkage.sh: the binary must load ITS OWN ggml (three-ggml-generations hazard)."""
    env = lib_env(build, "gpu" if build.name.startswith("gpu") else "cpu")
    r = subprocess.run([str(VERIFY), str(build / "bin" / exe), str(build)], env=env, capture_output=True, text=True)
    return {"build": str(build), "exe": exe, "rc": r.returncode, "tail": (r.stdout + r.stderr).strip().splitlines()[-3:]}

def strings_has(path: Path, needle: str) -> bool:
    r = subprocess.run(["strings", "-a", str(path)], capture_output=True, text=True)
    return needle in r.stdout

# ----------------------------------------------------------------------------------------------- process helpers (own children only)
def kill_verified(p: subprocess.Popen, log=None) -> bool:
    if p.poll() is not None: return True
    p.terminate()
    try: p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()
        try: p.wait(10)
        except subprocess.TimeoutExpired: pass
    ok = p.poll() is not None
    say(f"own pid {p.pid} {'confirmed dead' if ok else 'STILL ALIVE'}", log)
    return ok

def port_listening(port: int) -> bool:
    return subprocess.run(["ss", "-ltn", f"sport = :{port}"], capture_output=True, text=True).stdout.count("LISTEN") > 0

def port_owned_by(port: int, pid: int) -> bool:
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
    return f"pid={pid}," in out

def wait_own_port(port: int, p: subprocess.Popen, max_s: int = 1200) -> dict:
    """Health answers AND the LISTEN socket is owned by OUR pid (a foreign server once answered a health probe on this host)."""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < max_s:
        if p.poll() is not None: return {"ok": False, "why": f"server pid {p.pid} died rc={p.returncode}"}
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200 and port_owned_by(port, p.pid):
                return {"ok": True, "pid": p.pid, "port": port, "after_s": round(time.time() - t0, 1)}
        except Exception: pass
        time.sleep(3)
    return {"ok": False, "why": "timeout"}

def kfd_vram(pid: int) -> int:
    tot = 0
    for f in Path(f"/sys/class/kfd/kfd/proc/{pid}").glob("vram_*"):
        try: tot += int(f.read_text().strip())
        except Exception: pass
    return tot

def foreign_kfd_vram_mib(own: set[int] | None = None) -> tuple[int, list]:
    tot, rows = 0, []
    for d in Path("/sys/class/kfd/kfd/proc").glob("*"):
        try: pid = int(d.name)
        except ValueError: continue
        if own and pid in own: continue
        v = kfd_vram(pid); tot += v
        try: args = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()[:70]
        except Exception: args = "?"
        rows.append({"pid": pid, "vram_mib": v >> 20, "args": args})
    return tot >> 20, rows

class Residency:
    """Sample DURING the run (never after): KFD VRAM of OUR pid + libggml-hip mapped from the arm's own bin dir."""
    def __init__(self, pid: int, build: Path):
        self.pid, self.build, self.max_vram, self.n, self.hip_own, self.hip_foreign = pid, build, 0, 0, False, None
        self._stop = threading.Event(); self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
    def _run(self):
        while not self._stop.is_set():
            self.max_vram = max(self.max_vram, kfd_vram(self.pid)); self.n += 1
            try:
                for l in Path(f"/proc/{self.pid}/maps").read_text().splitlines():
                    if "libggml-hip" in l:
                        path = l.split()[-1]
                        if os.path.realpath(path).startswith(str((self.build / "bin").resolve())): self.hip_own = True
                        else: self.hip_foreign = path
            except Exception: pass
            self._stop.wait(1.0)
    def stop(self) -> dict:
        self._stop.set(); self._t.join(5)
        return {"pid": self.pid, "n_samples": self.n, "pid_vram_mib_max": self.max_vram >> 20, "hip_lib_from_own_build": self.hip_own,
                "hip_lib_foreign": self.hip_foreign, "resident": self.max_vram >= (1 << 30) and self.hip_own and not self.hip_foreign}

# ----------------------------------------------------------------------------------------------- parsers (unit-tested in --selftest)
def parse_tbo(text: str) -> dict:
    """test-backend-ops verdict. V11-086-DRV (2026-10-23): delegates to fa086v2_integrity.parse_tbo_text (ANSI-stripped, so the
    colour escape before FAIL no longer hides it; captures 'ERR|NMSE = x > thr' per-case records). Legacy keys are kept:
    max_nmse_on_fail is the max NMSE on a fail line (None if the build prints ERR instead) and max_err_on_fail covers ERR/NMSE."""
    r = parse_tbo_text(text)
    nm = [float(x) for x in re.findall(r"NMSE = ([0-9.eE+-]+)", _ANSI_RE.sub("", text))]
    return r | {"max_nmse_on_fail": max(nm) if nm else None}

PERF_RE = re.compile(r"MUL_MAT\((?P<vars>[^)]*)\):\s+(?P<runs>\d+) runs\s+-\s+(?P<us>[0-9.]+) us/run")
def parse_tbo_perf(text: str) -> dict:
    """{(type_a, n): us_per_run} from `test-backend-ops perf -o MUL_MAT -p type_a=...`. Keeps the mean of repeats per (type,n,m,k,bs)."""
    acc: dict = {}
    for l in text.splitlines():
        m = PERF_RE.search(l)
        if not m: continue
        v = dict(kv.split("=", 1) for kv in re.split(r",(?![^\[]*\])", m["vars"]) if "=" in kv)
        if v.get("type_b", "f32") != "f32" or v.get("bs", "[1,1]") != "[1,1]": continue
        key = (v["type_a"], int(v["n"]), v["m"], v["k"])
        acc.setdefault(key, []).append(float(m["us"]))
    return {k: statistics.mean(x) for k, x in acc.items()}

def parse_bench(text: str) -> list[dict]:
    """llama-bench -o json -> [{n_prompt,n_gen,n_depth,avg_ts,stddev_ts,test}] (JSON array on stdout; tolerate leading log noise)."""
    i = text.find("[")
    rows = json.loads(text[i:text.rfind("]") + 1]) if i >= 0 else []
    return [{"n_prompt": r["n_prompt"], "n_gen": r["n_gen"], "n_depth": r.get("n_depth", 0), "avg_ts": r["avg_ts"],
             "stddev_ts": r.get("stddev_ts"), "test": r.get("test")} for r in rows]

def cell_name(r: dict) -> str:
    base = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}"
    return base + (f"@d{r['n_depth']}" if r.get("n_depth") else "")

def med(x): return statistics.median(x)
def spread_pct(x):   # between-launch range / median, percent (unit: process-unit range of n launches)
    return (max(x) - min(x)) / med(x) * 100 if len(x) > 1 and med(x) else 0.0
def effect_pct(base, arm): return (med(arm) / med(base) - 1) * 100

def effects(cells: dict, base: str, arm: str) -> dict:
    """cells = {cell: {armname: [avg_ts per launch]}} -> {cell: {effect_pct, A_spread_pct, n}}"""
    out = {}
    for c, d in cells.items():
        if base in d and arm in d and d[base] and d[arm]:
            out[c] = {"effect_pct": round(effect_pct(d[base], d[arm]), 3), "base_spread_pct": round(spread_pct(d[base]), 3),
                      "arm_spread_pct": round(spread_pct(d[arm]), 3), "n_base": len(d[base]), "n_arm": len(d[arm])}
    return out

def floor_for(cell: str) -> float: return TG_FLOOR if cell.startswith("tg") else PP_FLOOR
def significant_gain(e: dict, cell: str) -> bool:
    """gain beyond max(ledger floor, base between-launch spread): the larger of the two units is the conservative bar."""
    return e["effect_pct"] > max(floor_for(cell), e["base_spread_pct"])
def regression(e: dict, cell: str) -> bool:
    return e["effect_pct"] < -floor_for(cell)

# ----------------------------------------------------------------------------------------------- decision rules (PREDECLARED.md mirrors these)
def serving_ab(v: dict | None) -> dict:
    """Digest of ab_probe summarize verdict: speed block + speed_no_regression + the PASS checks that are NOT identity (052/050 may change numerics)."""
    if not v: return {"present": False}
    sp = v.get("speed") or {}
    return {"present": True, "speed_no_regression": v.get("checks", {}).get("speed_no_regression"),
            "tg": sp.get("tg_tps_median"), "pp": sp.get("pp_tps_median"),
            "identity_C_equals_P": (v.get("identity") or {}).get("C_equals_P"),
            "residency_ok": all((v.get("gpu_resident_during_run") or {"x": False}).values()) and all((v.get("own_ggml_mapped") or {"x": False}).values())}

def decide_052(r: dict) -> dict:
    why = []
    tbo_ok = all(r["tbo"][a][op]["ok100"] for a in ("A", "B") for op in ("MUL_MAT", "MUL_MAT_ID"))
    if not tbo_ok: why.append("tbo MUL_MAT/MUL_MAT_ID not 100% on both arms")
    ab = serving_ab(r.get("serving_ab"))
    tg = ab.get("tg") if ab["present"] else None
    tg_ok = bool(tg) and tg["C_over_P_pct"] > max(TG_FLOOR, tg.get("noise_floor_pct") or 0)
    if not tg_ok: why.append("serving tg gain not beyond max(2.00%, P-spread) (or serving A/B absent)")
    guard = ab["present"] and ab.get("speed_no_regression") is True and ab.get("residency_ok")
    if not guard: why.append("serving A/B speed/residency guard not satisfied")
    eff = effects(r["bench"]["cells"], "A", "B")
    pp_cells = {c: e for c, e in eff.items() if c.startswith("pp")}
    pp_bad = {c: e["effect_pct"] for c, e in pp_cells.items() if regression(e, c)}
    pp_ok = bool(pp_cells) and not pp_bad
    if not pp_ok: why.append(f"pp widths regress beyond {PP_FLOOR}%: {pp_bad}" if pp_bad else "no pp cells measured")
    cut = lb1c_crossover(r.get("lb1c"))
    if tbo_ok and tg_ok and guard and pp_ok:
        dec = "fold_global_ne11_ge5"
    elif tbo_ok and tg_ok and guard and cut:
        dec = "fold_per_type"
    else:
        dec = "hold"
    return {"item": "V11-052", "decision": dec, "per_type_cuts": cut if dec == "fold_per_type" else {}, "pp_regressions": pp_bad,
            "tg_serving": tg, "reasons": why, "effects": eff}

def lb1c_crossover(lb1c: dict | None) -> dict:
    """lb1c = {type: {n: {"A": [us per launch], "B": [...]}}} for n 4..9.  cut_T = smallest n such that for EVERY m in [n,9] B beats A by more than
    max(0, A between-launch spread %).  Types with no such n are not routed."""
    cuts = {}
    for t, byn in (lb1c or {}).items():
        good = {}
        for n, d in byn.items():
            a, b = d.get("A") or [], d.get("B") or []
            if a and b: good[int(n)] = (med(a) / med(b) - 1) * 100 > spread_pct(a)
        ns = [n for n in range(4, 10)]
        cut = None
        for n in ns:
            if all(good.get(m, False) for m in range(n, 10)): cut = n; break
        if cut is not None: cuts[t] = cut
    return cuts

def decide_050(r: dict) -> dict:
    why = []
    b_suite_on_B, b_suite_on_A = r["tbo"]["B_suite_on_B"], r["tbo"].get("B_suite_on_A")
    # exactness: the arm is 100% on its own suite and passes at least the champion's pass count (2978 at C)
    exact_fa = bool(b_suite_on_B["ok100"] and (r["tbo"]["A_suite_on_A"]["passed"] or 0) <= (b_suite_on_B["passed"] or 0))
    finite = bool(r["long_prompt"]["B"]["finite_ok"])
    ident = r["greedy"]["identity"]               # "byte_identical" | "differs" | "not_measured"
    benefit_fails = None
    if b_suite_on_A and b_suite_on_A.get("total"):
        benefit_fails = (b_suite_on_A["total"] - (b_suite_on_A["passed"] or 0)) - (b_suite_on_B["total"] - (b_suite_on_B["passed"] or 0))
    benefit = bool(benefit_fails and benefit_fails > 0)
    eff = effects(r["bench"]["cells"], "A", "B")
    bad = {c: e["effect_pct"] for c, e in eff.items() if regression(e, c)}
    ab = serving_ab(r.get("serving_ab"))
    srv_ok = (not ab["present"]) or (ab.get("speed_no_regression") is True)
    perf_ok = bool(eff) and not bad and srv_ok
    if not exact_fa: why.append("tbo FLASH_ATTN_EXT not 100% on both arms")
    if not finite: why.append("long-prompt DFlash batch not finite on arm B")
    if not benefit: why.append("no stricter-exactness benefit shown (arm-suite failures on A-libs minus on B-libs <= 0 or not measurable)")
    if not perf_ok: why.append(f"perf regression beyond floors: {bad}" if bad else "perf cells missing / serving regression")
    if ident != "byte_identical": why.append(f"27B greedy identity {ident}")
    if exact_fa and finite and benefit and perf_ok and ident == "byte_identical": dec = "keep_default_on"
    elif exact_fa and finite and benefit: dec = "keep_opt_in"
    else: dec = "hold"
    return {"item": "V11-050", "decision": dec, "benefit_fail_delta": benefit_fails, "perf_regressions": bad, "reasons": why, "effects": eff}

def decide_051(r: dict) -> dict:
    why = []
    tbo_ok = all(r["tbo"][a]["ok100"] for a in ("A", "B", "B2"))
    ident = r["greedy"]["identity_A_vs_B"]
    eB, eB2 = effects(r["bench"]["cells"], "A", "B"), effects(r["bench"]["cells"], "A", "B2")
    shapes = [c for c in eB]
    gainB = [c for c in shapes if significant_gain(eB[c], c)]
    regB = {c: eB[c]["effect_pct"] for c in shapes if regression(eB[c], c)}
    gain_any = [c for c in shapes if significant_gain(eB[c], c) or (c in eB2 and significant_gain(eB2[c], c))]
    if not tbo_ok: why.append("tbo FLASH_ATTN_EXT not 100% on an env arm")
    if ident != "byte_identical": why.append(f"27B greedy identity A vs B: {ident}")
    if len(gainB) < 2: why.append(f"default cap-16 gains beyond floor at {len(gainB)} of {len(shapes)} shapes (<2)")
    if regB: why.append(f"default regresses beyond floor at {regB}")
    if tbo_ok and ident == "byte_identical" and len(gainB) >= 2 and not regB: dec = "keep_default_on"
    elif tbo_ok and gain_any: dec = "fold_opt_in_tune0_default"
    else: dec = "hold"
    return {"item": "V11-051", "decision": dec, "gain_shapes_default": gainB, "regressions_default": regB, "reasons": why,
            "effects_default": eB, "effects_max_cols_32": eB2}

def floor086(cells: dict, cell: str) -> dict:
    """floor for one cell, percent of median tok/s, process unit (n=3 launches/arm):
    max(MIN_FLOOR086, C between-launch range/median, |median(Bunset)/median(C)-1| = the identity-control effect)."""
    d = cells.get(cell) or {}
    if not d.get("C") or not d.get("Bunset"): return {"floor_pct": None}
    sp = spread_pct(d["C"]); idc = abs(effect_pct(d["C"], d["Bunset"]))
    return {"floor_pct": round(max(MIN_FLOOR086, sp, idc), 3), "C_spread_pct": round(sp, 3), "identity_control_pct": round(idc, 3)}

def decide_086(r: dict) -> dict:
    """V11-086 v2 rule (predeclared/086v2/PREDECLARED_086v2.md)."""
    why = []
    integ = r.get("integrity") or {}
    integrity_ok = all(integ.get(k) is True for k in ("tbo_fa_100", "fa_unset_bit_identical", "finite_all_dumps", "knob_strings_proof", "linkage_ok"))
    if not integrity_ok: why.append(f"integrity not all True: { {k: integ.get(k) for k in ('tbo_fa_100','fa_unset_bit_identical','finite_all_dumps','knob_strings_proof','linkage_ok')} }")
    cells = (r.get("bench") or {}).get("cells") or {}
    per = {}
    for c in CELLS086:
        d = cells.get(c) or {}
        if not all(len(d.get(a) or []) >= 3 for a in ("C", "Bunset", "Bauto")): why.append(f"cell {c} not measured n=3 on all arms"); continue
        f = floor086(cells, c); e = effect_pct(d["C"], d["Bauto"])
        per[c] = f | {"effect_pct": round(e, 3), "gain": e > f["floor_pct"], "regression": e < -f["floor_pct"]}
    all_cells = len(per) == len(CELLS086)
    sv = r.get("serving") or {}
    serving_ok = None
    if sv.get("effect_pct") is not None and "tg32" in per:
        serving_ok = sv["effect_pct"] >= -per["tg32"]["floor_pct"]
    else: why.append("serving gate (q38fn, REG=auto) not measured")
    long_gain = any(per.get(c, {}).get("gain") for c in LONG086)
    any_reg = any(v["regression"] for v in per.values())
    if not integrity_ok or not all_cells: dec = "hold"
    elif long_gain and not any_reg and serving_ok is True: dec = "default_on_VKQ_REG_auto"
    else: dec = "fold_opt_in_default_off"
    return {"item": "V11-086", "rule_version": "086v2", "decision": dec, "cells": per, "long_ctx_gain": long_gain, "any_regression": any_reg,
            "serving": sv, "serving_ok": serving_ok, "reasons": why}

DECIDE = {"052": decide_052, "050": decide_050, "051": decide_051, "086": decide_086}

# ----------------------------------------------------------------------------------------------- run plumbing
class Ctx:
    def __init__(self, item, run_dir, execute):
        self.item, self.dir, self.execute = item, Path(run_dir), execute
        self.log = self.dir / "driver.log"
        self.deadline = int(os.environ.get("FILLER_DEADLINE", "0")) or None
        self.stopfile = os.environ.get("FILLER_STOPFILE")
        self.pidfile = os.environ.get("FILLER_PIDFILE")
        self.res: dict = {}
        self.should_stop_flag = False
    def say(self, m): say(m, self.log)
    def time_left(self): return (self.deadline - time.time()) if self.deadline else 10 ** 6
    def should_stop(self):
        st = bool((self.stopfile and os.path.exists(self.stopfile)) or self.time_left() < 120)
        self.should_stop_flag |= st
        return st
    def save(self): (self.dir / "results.json").write_text(json.dumps(self.res, indent=1, default=str))
    def register(self, pid):
        if self.pidfile: open(self.pidfile, "a").write(f"{pid}\n")

def check_predeclared(item: str, run_dir: Path):
    f = PRE_DIR / PRE086[0] / PRE086[1] if item == "086" else PRE_DIR / item / "PREDECLARED.md"
    pins = dict(l.split()[::-1] for l in (PRE_DIR / "PINNED.sha256").read_text().splitlines() if l.strip()) if (PRE_DIR / "PINNED.sha256").exists() else {}
    key = f"{PRE086[0]}/{PRE086[1]}" if item == "086" else f"{item}/PREDECLARED.md"
    if not f.exists(): die(3, f"REFUSE: {f} missing (predeclare the rule before any run)")
    if pins.get(key) != sha256_file(f): die(3, f"REFUSE: {key} sha256 {sha256_file(f)[:12]} != pinned {str(pins.get(key))[:12]} (rule edited after pinning)")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "PREDECLARED.md").write_text(f.read_text())
    return sha256_file(f)

def inside_claim_or_die():
    if not (os.environ.get("FILLER_ID") or os.environ.get("V11_INSIDE_CLAIM") == "1"):
        die(3, "REFUSE: --execute only inside a claim (run via filler_run2.sh grant, or V11_INSIDE_CLAIM=1 under region-lock --gpu-quiet exclusive)")

def gpu_preflight(ctx: Ctx, ports=(8083, 18183, 8197), model_footprint_gib=50):
    bad = []
    for p in ports:
        if port_listening(p): bad.append(f"port {p} listening")
    tot, rows = foreign_kfd_vram_mib()
    limit = (64 - model_footprint_gib - 4) * 1024
    ctx.say(f"foreign KFD VRAM {tot} MiB (limit {limit} MiB); rows={rows}")
    if tot > limit: bad.append(f"foreign KFD VRAM {tot} MiB > {limit} MiB")
    return bad

def run_logged(argv, env, log: Path, timeout, cwd=None, ctx: Ctx | None = None):
    with open(log, "w") as f:
        p = subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=False)
        if ctx: ctx.register(p.pid)
        try: p.wait(timeout)
        except subprocess.TimeoutExpired:
            kill_verified(p, ctx.log if ctx else None); return 124, p.pid
        return p.returncode, p.pid

def tbo(ctx, name, exe_build: Path, lib_build: Path, dev: str, backend: str, op: str, extra_env=None, timeout=1500, perf_filter=None):
    t = int(min(timeout, max(0, ctx.time_left() - 200)))
    if t < 60: return {"skipped": "deadline"}
    log = ctx.dir / f"tbo_{name}.log"
    pre = ["taskset", "-c", GPU_THREADS, "numactl", "--membind=3", "--"] if dev == "gpu" else ["nice", "-n", "19"]
    argv = pre + [str(exe_build / "bin" / "test-backend-ops"), "perf" if perf_filter else "test", "-b", backend, "-o", op] + (["-p", perf_filter] if perf_filter else [])
    rc, pid = run_logged(argv, lib_env(lib_build, dev, extra_env), log, t, ctx=ctx)
    text = log.read_text(errors="replace")
    out = parse_tbo(text) | {"rc": rc, "log": str(log), "argv": argv}
    if perf_filter: out["perf"] = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in parse_tbo_perf(text).items()}
    ctx.say(f"tbo {name}: rc={rc} {out['passed']}/{out['total']} fails={out['n_fail_lines']}" if not perf_filter else f"tbo perf {name}: rc={rc} cases={len(out['perf'])}")
    return out

def gpu_bench_launch(ctx, tag, build: Path, model: Path, bench_args: list[str], env_extra=None, timeout=1500):
    """One llama-bench launch (own Popen), residency sampled DURING it, own-ggml proven by /proc/<pid>/maps. -o json."""
    t = int(min(timeout, max(0, ctx.time_left() - 200)))
    if t < 90: return {"skipped": "deadline"}
    argv = ["taskset", "-c", GPU_THREADS, "numactl", "--membind=3", "--", str(build / "bin" / "llama-bench"), "-m", str(model),
            "-ngl", "99", "-fa", "1", "-mmp", "0", "-t", "8", "-o", "json"] + bench_args
    out_f, err_f = ctx.dir / f"bench_{tag}.json", ctx.dir / f"bench_{tag}.err"
    with open(out_f, "w") as fo, open(err_f, "w") as fe:
        p = subprocess.Popen(argv, env=lib_env(build, "gpu", env_extra), stdout=fo, stderr=fe)
        ctx.register(p.pid); res = Residency(p.pid, build)
        try: p.wait(t)
        except subprocess.TimeoutExpired: kill_verified(p, ctx.log)
        r = res.stop()
    if p.returncode is None or p.returncode != 0:
        return {"error": f"llama-bench rc={p.returncode}", "residency": r, "err": err_f.read_text(errors="replace")[-400:]}
    rows = parse_bench(out_f.read_text())
    ctx.say(f"bench {tag}: {len(rows)} cells; vram_max={r['pid_vram_mib_max']} MiB resident={r['resident']}")
    return {"rows": rows, "residency": r, "argv": argv, "env_extra": env_extra or {}}

def aba_bench(ctx, key, arms: dict, model, bench_args, launches=3):
    """arms = {name: (build, env_extra)}; order A B (B2) A B (B2) ... `launches` times. Returns {cells, launches, residency}."""
    cells, launches_out, resid_ok = {}, [], True
    names = list(arms)
    for i in range(launches):
        for nm in names:
            if ctx.should_stop(): ctx.say(f"stop/deadline before {key} launch {i}/{nm}"); return {"cells": cells, "launches": launches_out, "complete": False, "residency_ok": resid_ok}
            b, env = arms[nm]
            r = gpu_bench_launch(ctx, f"{key}-{nm}{i}", b, model, bench_args, env)
            launches_out.append({"arm": nm, "i": i, **{k: v for k, v in r.items() if k != "rows"}})
            if "rows" not in r: ctx.say(f"{key} {nm}{i}: {r}"); resid_ok = False; continue
            resid_ok &= bool(r["residency"]["resident"])
            for row in r["rows"]:
                cells.setdefault(cell_name(row), {}).setdefault(nm, []).append(row["avg_ts"])
    return {"cells": cells, "launches": launches_out, "complete": True, "residency_ok": resid_ok}

# ------------------------------------------------ 27B greedy server (argv from the live 8083 record; own pid; port 18183)
GREEDY_PROMPTS = ["The three main reasons the EPYC memory hierarchy has an L3 cache are",
                  "Write a Python function that merges two sorted lists into one sorted list, and explain its complexity.",
                  "Summarize the trade-offs between speculative decoding with a small draft model and with a learned drafter head."]
def long_prompt() -> str:
    src = Path("/mnt/raid0/llm/llama.cpp/ggml/src/ggml-cuda/mmq.cuh")   # read-only, stable text; ~30 KB ~ 8-10k tokens
    return "Review the following CUDA source and list its three riskiest numerical assumptions.\n\n" + src.read_text(errors="replace")[:30000]

def server_argv(build: Path, port: int, slots: Path) -> tuple[list[str], dict]:
    d = json.loads(REC8083.read_text())
    a = d["argv"] if isinstance(d["argv"], list) else shlex.split(d["argv"])
    a = [str(build / "bin" / "llama-server") if t.endswith("/bin/llama-server") else t for t in a]
    if "--port" not in a or "--slot-save-path" not in a: die(3, "8083 record lacks --port/--slot-save-path")
    a[a.index("--port") + 1] = str(port); a[a.index("--slot-save-path") + 1] = str(slots)
    ld = [str(build / "bin")] + [p for p in d["ld_library_path"].split(":") if "/kernels/builds/" not in p]
    return a, {"LD_LIBRARY_PATH": ":".join(ld)}

def greedy_arm(ctx, name, build: Path, env_extra=None, port=18183, with_long=False):
    import urllib.request
    slots = ctx.dir / "slots"; slots.mkdir(exist_ok=True)
    argv, envd = server_argv(build, port, slots)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")} | envd | (env_extra or {})
    if port_listening(port): return {"error": f"port {port} already listening (not ours)"}
    slog = ctx.dir / f"server-{name}.log"
    with open(slog, "w") as f: p = subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    ctx.register(p.pid)
    out = {"argv": argv, "env_extra": env_extra or {}, "pid": p.pid, "prompts": {}}
    try:
        h = wait_own_port(port, p, 900); out["port_proof"] = h
        if not h["ok"]: return out | {"error": h.get("why")}
        res = Residency(p.pid, build)
        props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10).read()); out["build_info"] = props.get("build_info")
        prompts = list(GREEDY_PROMPTS) + ([long_prompt()] if with_long else [])
        for k, pr in enumerate(prompts, 1):
            body = json.dumps({"prompt": pr, "n_predict": 128, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                               "ignore_eos": True, "return_tokens": True}).encode()
            try:
                r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"}), timeout=900).read())
                out["prompts"][f"p{k}"] = {"content": r.get("content"), "tokens": r.get("tokens"), "n_predicted": (r.get("timings") or {}).get("predicted_n"),
                                           "draft_n_accepted": (r.get("timings") or {}).get("draft_n_accepted"), "long": k > len(GREEDY_PROMPTS)}
            except Exception as e: out["prompts"][f"p{k}"] = {"error": str(e)}
        out["residency"] = res.stop()
    finally:
        out["dead"] = kill_verified(p, ctx.log)
    out["log_nonfinite_hits"] = len(re.findall(r"non-?finite|\bnan\b|\binf\b", slog.read_text(errors="replace"), re.I))
    return out

def greedy_identity(x: dict, y: dict) -> str:
    ks = [k for k in x.get("prompts", {}) if not x["prompts"][k].get("long")]
    if not ks or any("error" in x["prompts"][k] or "error" in y.get("prompts", {}).get(k, {"error": 1}) for k in ks): return "not_measured"
    return "byte_identical" if all(x["prompts"][k]["content"] == y["prompts"][k]["content"] and x["prompts"][k]["tokens"] == y["prompts"][k]["tokens"] for k in ks) else "differs"

def long_ok(arm: dict) -> bool:
    ks = [k for k, v in arm.get("prompts", {}).items() if v.get("long")]
    return bool(ks) and all(not arm["prompts"][k].get("error") and (arm["prompts"][k].get("n_predicted") or 0) >= 100 for k in ks) and arm.get("log_nonfinite_hits", 1) == 0

def gab_path() -> Path:
    for p in GAB_CANDIDATES:
        if p.exists(): return p
    die(5, "build-missing: gpu_champion_ab.sh not found")

def run_gpu_ab(ctx, p_build: Path, c_build: Path, tag: str, dry=False):
    gab = gab_path()
    args = [str(gab), "--p-build", str(p_build), "--p-tip", build_sha(p_build), "--p-buildno", os.environ.get("P_BUILDNO", "0"),
            "--c-build", str(c_build), "--c-tip", build_sha(c_build), "--c-buildno", os.environ.get("C_BUILDNO", "0"), "--run", str(ctx.dir / f"gpu_ab_{tag}")]
    if dry: return {"argv": args + ["--dry-run"], "dry": True}
    for b in (p_build, c_build):                                   # build numbers from the binaries themselves (loads HIP libs; inside the claim only)
        r = subprocess.run([str(b / "bin" / "llama-server"), "--version"], env=lib_env(b, "gpu"), capture_output=True, text=True)
        m = re.search(r"version:\s*(\d+)", r.stdout + r.stderr)
        args[args.index("--p-buildno" if b == p_build else "--c-buildno") + 1] = m.group(1) if m else "0"
    t = int(max(0, ctx.time_left() - 240))
    if t < 2400: return {"skipped": f"deadline: {t}s left, 15 pairs need ~40 min"}
    env = dict(os.environ) | {"AB_INSIDE_CLAIM": "1"}
    rc, pid = run_logged(args + ["--execute"], env, ctx.dir / f"gpu_ab_{tag}.log", t, ctx=ctx)
    rj = ctx.dir / f"gpu_ab_{tag}" / "result.json"
    return {"rc": rc, "verdict": json.loads(rj.read_text())["verdict"] if rj.exists() else None, "result": str(rj)}

# ----------------------------------------------------------------------------------------------- items
BENCH_MAIN_052 = ["-p", "4,5,6,8,9,16", "-n", "128", "-r", "3"]
def run_052(ctx: Ctx, dry: bool):
    A, B = require_c("gpu"), arm_build("052")
    ctx.res["builds"] = {"A": str(A), "B": str(B), "linkage": [linkage_proof(A), linkage_proof(B)]}
    if any(x["rc"] for x in ctx.res["builds"]["linkage"]): die(7, f"linkage proof failed {ctx.res['builds']['linkage']}")
    if dry: return plan_gpu(ctx, [("tbo", "MUL_MAT,MUL_MAT_ID per arm"), ("bench", BENCH_MAIN_052), ("deep", "-p 5,8 -d 49152 -r 1"), ("lb1c", "tbo perf type sweep"), ("gpu_ab", "15 pairs")], A, B)
    ctx.res["tbo"] = {a: {op: tbo(ctx, f"{a}_{op}", b, b, "gpu", "ROCm0", op) for op in ("MUL_MAT", "MUL_MAT_ID")} for a, b in (("A", A), ("B", B))}
    arms = {"A": (A, None), "B": (B, None)}
    main = aba_bench(ctx, "main", arms, M27_Q4, BENCH_MAIN_052 + ["-d", "0,2048"])
    deep = aba_bench(ctx, "deep", arms, M27_Q4, ["-p", "5,8", "-n", "0", "-d", "49152", "-r", "1"])
    cells = main["cells"] | deep["cells"]
    ctx.res["bench"] = {"cells": cells, "main": main, "deep": deep, "model": str(M27_Q4)}
    ctx.res["lb1c"] = run_lb1c(ctx, A, B)
    ctx.res["serving_ab"] = (run_gpu_ab(ctx, A, B, "052") or {}).get("verdict")
    ctx.save()

LB1C_TYPES = ["q4_K", "q6_K", "iq4_xs", "iq4_nl", "iq3_s"]
def run_lb1c(ctx, A, B, launches=3):
    """EXL3-LB1c per-type ne11 4..9 sweep on MUL_MAT cases via tbo perf (both arms alternating). Model-independent: the served ne00/ne01 shapes are in tbo's case list."""
    out = {}
    for t in LB1C_TYPES:
        byn: dict = {}
        for i in range(launches):
            for nm, b in (("A", A), ("B", B)):
                if ctx.should_stop(): break
                r = tbo(ctx, f"lb1c_{t}_{nm}{i}", b, b, "gpu", "ROCm0", "MUL_MAT", timeout=600, perf_filter=f"type_a={t},type_b=f32")
                for k, us in (r.get("perf") or {}).items():
                    ty, n, m, kk = k.split("|")
                    if 4 <= int(n) <= 9: byn.setdefault((int(n), m, kk), {}).setdefault(nm, []).append(us)
        # collapse shapes: per n, per launch, mean over served shapes
        per_n: dict = {}
        for (n, m, kk), d in byn.items():
            for nm, v in d.items():
                per_n.setdefault(n, {}).setdefault(nm, []).append(v)
        out[t] = {n: {nm: [statistics.mean(col) for col in zip(*v)] for nm, v in d.items() if v and all(len(x) == len(v[0]) for x in v)} for n, d in per_n.items()}
    return out

def run_050(ctx: Ctx, dry: bool):
    A, B = require_c("gpu"), arm_build("050")
    ctx.res["builds"] = {"A": str(A), "B": str(B), "linkage": [linkage_proof(A), linkage_proof(B)]}
    if any(x["rc"] for x in ctx.res["builds"]["linkage"]): die(7, f"linkage proof failed {ctx.res['builds']['linkage']}")
    if dry: return plan_gpu(ctx, [("tbo", "FLASH_ATTN_EXT: A suite on A, B suite on A-libs and B-libs"), ("greedy", "27B DFlash2 A vs B + long prompt on B"),
                                  ("bench", ["-p", "512,2048", "-n", "128", "-r", "3"]), ("gpu_ab", "15 pairs")], A, B)
    FA = "FLASH_ATTN_EXT"
    ctx.res["tbo"] = {"A_suite_on_A": tbo(ctx, "A_on_A", A, A, "gpu", "ROCm0", FA),
                      "B_suite_on_B": tbo(ctx, "B_on_B", B, B, "gpu", "ROCm0", FA),
                      "B_suite_on_A": tbo(ctx, "Bsuite_on_Alibs", B, A, "gpu", "ROCm0", FA)}
    ga, gb = greedy_arm(ctx, "A", A), greedy_arm(ctx, "B", B, with_long=True)
    ctx.res["greedy"] = {"identity": greedy_identity(ga, gb), "A": ga, "B": gb}
    ctx.res["long_prompt"] = {"B": {"finite_ok": long_ok(gb), "nonfinite_log_hits": gb.get("log_nonfinite_hits")}}
    ctx.res["bench"] = aba_bench(ctx, "main", {"A": (A, None), "B": (B, None)}, M27_Q8, ["-p", "512,2048", "-n", "128", "-r", "3"])
    ctx.res["serving_ab"] = (run_gpu_ab(ctx, A, B, "050") or {}).get("verdict")
    ctx.save()

D256_ENV = {"A": {"GGML_CUDA_FA_CDNA_D256_TUNE": "0"}, "B": {}, "B2": {"GGML_CUDA_FA_CDNA_D256_MAX_COLS": "32"}}
def run_051(ctx: Ctx, dry: bool):
    B = arm_build("051")
    lib = B / "bin" / "libggml-hip.so"
    libs = sorted((B / "bin").glob("libggml-hip.so*"))
    for knob in ("GGML_CUDA_FA_CDNA_D256_MAX_COLS", "GGML_CUDA_FA_CDNA_D256_TUNE"):
        if STANDIN: break
        if not libs or not any(strings_has(l, knob) for l in libs if l.is_file() and not l.is_symlink()):
            die(5, f"build-missing: knob {knob} not compiled into {B}/bin/libggml-hip.so* (strings proof)")
    ctx.res["builds"] = {"B": str(B), "knob_strings_proof": True, "linkage": [linkage_proof(B)]}
    if ctx.res["builds"]["linkage"][0]["rc"]: die(7, "linkage proof failed")
    if dry: return plan_gpu(ctx, [("tbo", "FLASH_ATTN_EXT x3 env arms"), ("greedy", "27B DFlash2 TUNE=0 vs default (one binary)"),
                                  ("bench", ["-p", "128,512,2048", "-n", "128", "-r", "3"])], B, B)
    ctx.res["tbo"] = {a: tbo(ctx, f"env{a}", B, B, "gpu", "ROCm0", "FLASH_ATTN_EXT", D256_ENV[a]) for a in ("A", "B", "B2")}
    ga, gb = greedy_arm(ctx, "A", B, D256_ENV["A"]), greedy_arm(ctx, "B", B, D256_ENV["B"])
    ctx.res["greedy"] = {"identity_A_vs_B": greedy_identity(ga, gb), "A": ga, "B": gb}
    ctx.res["bench"] = aba_bench(ctx, "main", {n: (B, D256_ENV[n]) for n in ("A", "B", "B2")}, M27_Q8, ["-p", "128,512,2048", "-n", "128", "-r", "3"])
    ctx.save()

def plan_gpu(ctx, steps, A, B):
    argv, envd = server_argv(B, 18183, ctx.dir / "slots")
    bad = gpu_preflight(ctx)
    ctx.res["plan"] = {"steps": steps, "server_argv": argv, "server_ld": envd, "preflight_would_refuse": bad,
                       "gpu_ab": run_gpu_ab(ctx, A, B, ctx.item, dry=True) if ctx.item in ("052", "050") else None,
                       "bench_example": ["taskset", "-c", GPU_THREADS, "numactl", "--membind=3", "--", str(B / "bin" / "llama-bench"), "-m", "<model>", "-ngl", "99", "-fa", "1", "-mmp", "0", "-t", "8", "-o", "json"]}
    for k in ("M27_Q8", "M27_Q4"):
        if not globals()[k].exists(): ctx.res["plan"].setdefault("missing_models", []).append(str(globals()[k]))
    ctx.say("PLAN " + json.dumps(ctx.res["plan"], default=str)[:1500])
    ctx.save()

# ---- 086 v2 (CPU, Qwen3.8-Flash-Next)
GATE_PORTS = (18651,)          # the q38fn serving gate's port (q38fn-cpu-t48-mtp-d4.launch.json)

def foreign_llama_servers(blocking_only: bool = True) -> list[dict]:
    """Observation only (/proc; no signalling): any llama-server / llama-bench / llama-cli process not ours. The window is whole-host quiet."""
    hits = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit(): continue
        try: cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, ProcessLookupError): continue
        except Exception as e: hits.append({"pid": int(d.name), "cmd": f"<unreadable: {type(e).__name__}>"}); continue
        exe = cmd.split(" ", 1)[0]
        if not exe.endswith(("/llama-server", "/llama-bench", "/llama-cli", "/llama-perplexity")): continue
        # GPU-resident runners and embedders are the grant owner's call (the window is granted whole-host quiet by workspace-58);
        # they are recorded, not refused. Only foreign CPU inference refuses / skips a launch.
        gpu = "ROCm" in cmd or " -ngl all" in cmd or " -ngl 99" in cmd
        emb = "--embedding" in cmd or "--embeddings" in cmd
        hits.append({"pid": int(d.name), "cmd": cmd[:140], "blocking": not (gpu or emb)})
    return [h for h in hits if h["blocking"]] if blocking_only else hits

def cpu_preflight(ctx):
    bad = []
    nb = [h for h in foreign_llama_servers(blocking_only=False) if not h["blocking"]]
    if nb: ctx.say(f"non-blocking foreign processes (GPU/embedding, recorded): {[(h['pid'], h['cmd'][:80]) for h in nb]}"); ctx.res["foreign_nonblocking"] = nb
    for h in foreign_llama_servers(): bad.append(f"foreign CPU inference process pid {h['pid']}: {h['cmd']}")
    for p in GATE_PORTS:
        if port_listening(p): bad.append(f"port {p} busy")
    free = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) >> 20
    ctx.say(f"MemAvailable {free} GiB (086v2 needs 95 GiB = weights 87.3 + MTP head 2.6 + KV/state ~1 + 4)")
    if free < 100: bad.append(f"MemAvailable {free} GiB < 100")
    return bad

class OwnLibProof:
    """Samples /proc/<pid>/maps DURING the run: libggml-cpu must map from the arm's own bin dir (three-ggml-generations hazard)."""
    def __init__(self, pid: int, build: Path):
        self.pid, self.own, self.seen, self.n = pid, str(build / "bin") + "/", set(), 0
        self._stop = threading.Event(); self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
    def _run(self):
        while not self._stop.is_set():
            try:
                for l in Path(f"/proc/{self.pid}/maps").read_text().splitlines():
                    if "libggml" in l: self.seen.add(l.split()[-1])
                self.n += 1
            except Exception: pass
            self._stop.wait(5)
    def stop(self) -> dict:
        self._stop.set(); self._t.join(5)
        cpu = sorted(x for x in self.seen if "libggml-cpu" in x)
        return {"n_samples": self.n, "libggml_cpu": cpu, "own": bool(cpu) and all(x.startswith(self.own) for x in self.seen)}

def cpu_bench_launch(ctx, tag, build: Path, env_extra=None, timeout=3000):
    t = int(min(timeout, max(0, ctx.time_left() - 200)))
    if t < 600: return {"skipped": "deadline"}
    argv = Q38_PREFIX + [str(build / "bin" / "llama-bench"), "-m", str(Q38_TRUNK)] + Q38_BENCH_ARGS
    env = lib_env(build, "cpu", Q38_GGML_ENV | (env_extra or {}))
    out_f, err_f = ctx.dir / f"bench_{tag}.json", ctx.dir / f"bench_{tag}.err"
    with open(out_f, "w") as fo, open(err_f, "w") as fe:
        p = subprocess.Popen(argv, env=env, stdout=fo, stderr=fe)
        ctx.register(p.pid); proof = OwnLibProof(p.pid, build)
        try: p.wait(t)
        except subprocess.TimeoutExpired: kill_verified(p, ctx.log)
        lp = proof.stop()
    if p.returncode != 0:
        return {"error": f"llama-bench rc={p.returncode}", "own_lib": lp, "err": err_f.read_text(errors="replace")[-400:]}
    text = out_f.read_text()
    try:
        protocol = validate_bench_protocol(text, ["-m", str(Q38_TRUNK)] + Q38_BENCH_ARGS)
    except BenchProtocolError as exc:
        ctx.say(f"bench {tag}: protocol refused: {exc}")
        return {"error": f"bench protocol refused: {exc}", "own_lib": lp, "argv": argv}
    rows = parse_bench(text)
    ctx.say(f"bench {tag}: {len(rows)} cells own_lib={lp['own']} " + " ".join(f"{cell_name(r)}={r['avg_ts']:.2f}" for r in rows))
    return {"rows": rows, "protocol": protocol, "own_lib": lp, "argv": argv, "env_extra": env_extra or {}}

def run_086(ctx: Ctx, dry: bool):
    C, B = require_c("cpu"), arm_build("086")
    libs = [l for l in sorted((B / "bin").glob("libggml-cpu.so*")) if l.is_file() and not l.is_symlink()]
    knob = bool(libs) and any(strings_has(l, "GGML_FA_VKQ_REG") for l in libs)
    if not knob: die(5, f"build-missing: GGML_FA_VKQ_REG not compiled into {B}/bin/libggml-cpu.so (strings proof)")
    if any(strings_has(l, "GGML_FA_VKQ_REG") for l in sorted((C / "bin").glob("libggml-cpu.so*")) if l.is_file() and not l.is_symlink()):
        die(7, f"anchor C {C} unexpectedly carries GGML_FA_VKQ_REG (wrong anchor)")
    if not (CORR086 / "INTEGRITY.json").exists(): die(5, f"build-missing: {CORR086}/INTEGRITY.json (correctness half of v11-086-fa-vkq-build not finished)")
    link = [linkage_proof(C, "llama-bench"), linkage_proof(B, "llama-bench")]
    integ = json.loads((CORR086 / "INTEGRITY.json").read_text())
    ctx.res["builds"] = {"C": str(C), "B": str(B), "knob_strings_proof": knob, "linkage": link}
    ctx.res["integrity"] = {k: integ.get(k) for k in ("tbo_fa_100", "fa_unset_bit_identical", "finite_all_dumps", "knob_strings_proof")} | \
        {"linkage_ok": all(x["rc"] == 0 for x in link), "source": str(CORR086 / "INTEGRITY.json")}
    if not ctx.res["integrity"]["linkage_ok"]: die(7, f"linkage proof failed: {link}")
    if not Q38_TRUNK.exists(): die(5, f"model missing: {Q38_TRUNK}")
    arms = {"C": (C, {}), "Bunset": (B, {}), "Bauto": (B, {"GGML_FA_VKQ_REG": "auto"})}
    gc = HERE / "gate_compare_env.py"
    gate_cmd = [sys.executable, str(gc), "q38fn", "--anchor-build", str(C), "--cand-build", str(B), "--cand-env", "GGML_FA_VKQ_REG=auto",
                "--pairs", "5", "--out", str(ctx.dir / "gate_q38fn_auto")]
    ctx.res["plan"] = {"bench_argv": Q38_PREFIX + ["<build>/bin/llama-bench", "-m", str(Q38_TRUNK)] + Q38_BENCH_ARGS,
                       "bench_env": Q38_GGML_ENV, "launch_order": [list(o) for o in ROT086], "gate_cmd": gate_cmd + ["--execute"],
                       "integrity_now": ctx.res["integrity"], "preflight_would_refuse": cpu_preflight(ctx)}
    if dry:
        r = subprocess.run(gate_cmd + ["--dry-run"], capture_output=True, text=True, timeout=300)
        ctx.say(f"gate dry q38fn auto rc={r.returncode}: {(r.stdout.strip().splitlines() or r.stderr.strip().splitlines() or ['<no output>'])[-1][:200]}")
        ctx.res["plan"]["gate_dry_rc"] = r.returncode
        ctx.save(); return
    cells, launches = {}, []
    for i, order in enumerate(ROT086):
        for nm in order:
            if ctx.should_stop(): ctx.say(f"stop/deadline before launch {i}/{nm}"); launches.append({"arm": nm, "i": i, "skipped": "deadline"}); continue
            if foreign_llama_servers(): ctx.say("foreign inference process appeared; launch skipped"); launches.append({"arm": nm, "i": i, "skipped": "foreign-process"}); continue
            b, env = arms[nm]
            r = cpu_bench_launch(ctx, f"{nm}{i}", b, env)
            launches.append({"arm": nm, "i": i, **{k: v for k, v in r.items() if k != "rows"}})
            if "rows" not in r: ctx.say(f"launch {nm}{i}: {r}"); continue
            if not r["own_lib"]["own"]: ctx.say(f"launch {nm}{i}: own-lib proof FAILED {r['own_lib']}; launch discarded"); continue
            for row in r["rows"]: cells.setdefault(cell_name(row), {}).setdefault(nm, []).append(row["avg_ts"])
            ctx.res["bench"] = {"cells": cells, "launches": launches}; ctx.save()
    ctx.res["bench"] = {"cells": cells, "launches": launches, "complete": len([l for l in launches if "skipped" not in l and "error" not in l]) == 9}
    ctx.save()
    if ctx.should_stop() or foreign_llama_servers(): ctx.say("serving gate skipped (deadline/foreign process)"); ctx.res["serving"] = {"skipped": "deadline-or-foreign"}; ctx.save(); return
    rc, _ = run_logged(gate_cmd + ["--execute"], {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")}, ctx.dir / "gate_q38fn_auto.log",
                       int(max(60, ctx.time_left() - 200)), ctx=ctx)
    f = ctx.dir / "gate_q38fn_auto" / "compare-q38fn.json"
    if rc == 0 and f.exists():
        c = json.loads(f.read_text())["comparison"]
        ctx.res["serving"] = {"effect_pct": c["effect_pct"], "anchor_tok_s": c.get("anchor_tok_s"), "candidate_tok_s": c.get("candidate_tok_s"),
                              "unit": "process (matched_process_v2, 5 pairs), sign claim; no calibrated floor"}
    else:
        ctx.say(f"gate q38fn/auto rc={rc} (no compare json)"); ctx.res["serving"] = {"error": f"rc={rc}"}
    ctx.save()

RUNNERS = {"052": run_052, "050": run_050, "051": run_051, "086": run_086}

def has_skip(o) -> bool:
    if isinstance(o, dict):
        return any(k == "skipped" or (k == "complete" and v is False) or has_skip(v) for k, v in o.items())
    if isinstance(o, list): return any(has_skip(x) for x in o)
    return False

def write_decision(ctx: Ctx):
    complete = not has_skip(ctx.res) and not ctx.should_stop_flag
    (ctx.dir / ("COMPLETE" if complete else "INCOMPLETE")).write_text(now_iso() + "\n")
    d = {"complete": complete} | DECIDE[ctx.item](ctx.res) | {"decided_at": now_iso(), "rule": (f"predeclared/{PRE086[0]}/{PRE086[1]}" if ctx.item == "086" else f"predeclared/{ctx.item}/PREDECLARED.md"), "predeclared_sha256": ctx.res.get("predeclared_sha256")}
    (ctx.dir / "DECISION.json").write_text(json.dumps(d, indent=1, default=str))
    ctx.say(f"decision={d['decision']}")
    return d

# ----------------------------------------------------------------------------------------------- selftest
def selftest():
    n = 0
    def ok(c, m):
        nonlocal n; n += 1
        if not c: print("SELFTEST FAIL:", m); sys.exit(1)
    ok(parse_tbo("Backend 1/2: ROCm0\n  1145/1145 tests passed\n")["ok100"], "tbo pass")
    t = parse_tbo("  FLASH_ATTN_EXT(x): [FLASH_ATTN_EXT] NMSE = 0.000000103 > 0.000000100 FAIL\n  2977/2978 tests passed\n")
    ok(not t["ok100"] and t["max_nmse_on_fail"] == 1.03e-7, "tbo fail")
    # V11-086-DRV: captured ANSI-coloured ERR fail lines (LONGFA.json first_fails) must register as failures, same as parse_tbo_text
    _red, _lf = "\x1b[1;31mFAIL\x1b[0m", "/mnt/raid0/llm/tmp/v11-runs/086v2/longfa/LONGFA.json"
    try: _ff = json.load(open(_lf))["arms"]["long_C"]["first_fails"]
    except OSError: _ff = ["[FLASH_ATTN_EXT] ERR = 0.001441504 > 0.000500000   FLASH_ATTN_EXT(hsk=256,hsv=256,kv=65536,nb=1): test failed FAIL"]
    _txt = "\n".join(l.replace("FAIL", _red) for l in _ff) + "\n  31/35 tests passed\n"
    _d, _i = parse_tbo(_txt), parse_tbo_text(_txt)
    ok(not _d["ok100"] and _d["n_fail_lines"] == _i["n_fail_lines"] >= 4 and _d["passed"] == 31 and _d["max_err_on_fail"] == _i["max_err_on_fail"] > 5e-4, "tbo ANSI ERR fails parsed like parse_tbo_text")
    pt = parse_tbo_perf("  MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=5,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):   1000 runs -   91.31 us/run -     8192 flops/run -  5 GFLOPS\n")
    ok(pt == {("q4_K", 5, "4096", "14336"): 91.31}, f"perf parse {pt}")
    rows = parse_bench('noise\n[{"n_prompt":5,"n_gen":0,"n_depth":2048,"avg_ts":100.0,"stddev_ts":1.0,"test":"pp5 @ d2048"},{"n_prompt":0,"n_gen":128,"avg_ts":40.0}]\n')
    ok([cell_name(r) for r in rows] == ["pp5@d2048", "tg128"], "cell names")
    mk = lambda A, B: {"A": A, "B": B}
    base_ok = {"MUL_MAT": {"ok100": True}, "MUL_MAT_ID": {"ok100": True}}
    cellsok = {"pp4": mk([100, 101, 99], [95, 96, 95]), "pp8": mk([100, 100, 100], [120, 120, 121]), "tg128": mk([40, 40, 40], [40, 40, 40])}
    tgv = {"checks": {"speed_no_regression": True}, "speed": {"tg_tps_median": {"C_over_P_pct": 30.0, "noise_floor_pct": 1.0}}, "gpu_resident_during_run": {"1": True}, "own_ggml_mapped": {"1": True}}
    r = {"tbo": {"A": base_ok, "B": base_ok}, "bench": {"cells": cellsok}, "serving_ab": tgv, "lb1c": {}}
    ok(decide_052(r)["decision"] == "fold_global_ne11_ge5", "052 global")
    r2 = json.loads(json.dumps(r)); r2["bench"]["cells"]["pp4"] = mk([100, 100, 100], [68, 68, 68])
    ok(decide_052(r2)["decision"] == "hold", "052 pp regress w/o crossover -> hold")
    r2["lb1c"] = {"q4_K": {str(n): {"A": [100, 101, 100], "B": [60, 61, 60]} for n in range(4, 10)}, "q6_K": {str(n): {"A": [100, 100, 100], "B": [100, 99, 100]} for n in range(4, 10)}}
    d = decide_052(r2); ok(d["decision"] == "fold_per_type" and d["per_type_cuts"] == {"q4_K": 4}, f"052 per-type {d['per_type_cuts']}")
    r3 = json.loads(json.dumps(r)); r3["serving_ab"]["speed"]["tg_tps_median"]["C_over_P_pct"] = 1.0
    ok(decide_052(r3)["decision"] == "hold", "052 tg below floor -> hold")
    r4 = json.loads(json.dumps(r)); r4["tbo"]["B"]["MUL_MAT"]["ok100"] = False
    ok(decide_052(r4)["decision"] == "hold", "052 tbo fail -> hold")
    ok(lb1c_crossover({"t": {"4": {"A": [10, 10], "B": [20, 20]}, "5": {"A": [10, 10], "B": [5, 5]}, "6": {"A": [10, 10], "B": [5, 5]}, "7": {"A": [10, 10], "B": [5, 5]}, "8": {"A": [10, 10], "B": [5, 5]}, "9": {"A": [10, 10], "B": [5, 5]}}}) == {"t": 5}, "crossover 5")
    t100 = {"ok100": True, "passed": 10, "total": 10}
    r50 = {"tbo": {"A_suite_on_A": t100, "B_suite_on_B": t100, "B_suite_on_A": {"passed": 8, "total": 10, "ok100": False}}, "long_prompt": {"B": {"finite_ok": True}},
           "greedy": {"identity": "byte_identical"}, "bench": {"cells": {"pp512": mk([100] * 3, [98] * 3), "tg128": mk([40] * 3, [40] * 3)}}, "serving_ab": None}
    ok(decide_050(r50)["decision"] == "keep_default_on", "050 default-on")
    r50b = json.loads(json.dumps(r50)); r50b["bench"]["cells"]["pp512"] = mk([100] * 3, [80] * 3)
    ok(decide_050(r50b)["decision"] == "keep_opt_in", "050 perf cost -> opt-in")
    r50c = json.loads(json.dumps(r50)); r50c["tbo"]["B_suite_on_A"] = t100
    ok(decide_050(r50c)["decision"] == "hold", "050 no benefit -> hold")
    r50d = json.loads(json.dumps(r50)); r50d["greedy"]["identity"] = "differs"
    ok(decide_050(r50d)["decision"] == "keep_opt_in", "050 differs -> opt-in")
    cells51 = {"pp128": mk([100] * 3, [101] * 3), "pp512": mk([100] * 3, [115] * 3), "pp2048": mk([100] * 3, [112] * 3), "tg128": mk([40] * 3, [40] * 3)}
    for c in cells51.values(): c["B2"] = [100] * 3
    r51 = {"tbo": {"A": t100, "B": t100, "B2": t100}, "greedy": {"identity_A_vs_B": "byte_identical"}, "bench": {"cells": cells51}}
    ok(decide_051(r51)["decision"] == "keep_default_on", "051 default-on")
    r51b = json.loads(json.dumps(r51)); r51b["bench"]["cells"]["pp2048"] = mk([100] * 3, [101] * 3)
    ok(decide_051(r51b)["decision"] == "fold_opt_in_tune0_default", "051 one gain -> opt-in")
    r51c = json.loads(json.dumps(r51)); r51c["bench"]["cells"] = {c: mk([100] * 3, [100] * 3) for c in cells51}
    ok(decide_051(r51c)["decision"] == "hold", "051 no gain -> hold")
    ok(has_skip({"a": [{"skipped": "deadline"}]}) and has_skip({"b": {"complete": False}}) and not has_skip({"b": {"complete": True}}), "has_skip")
    ok_i = {"tbo_fa_100": True, "fa_unset_bit_identical": True, "finite_all_dumps": True, "knob_strings_proof": True, "linkage_ok": True}
    def r86(long_eff=1.06, short_eff=1.0, idc=1.0, sv=0.5, integ=ok_i):
        cells = {}
        for c in CELLS086:
            e = long_eff if c in LONG086 else short_eff
            cells[c] = {"C": [100.0, 100.5, 99.8], "Bunset": [100.0 * idc] * 3, "Bauto": [100.0 * e, 100.2 * e, 99.9 * e]}
        return {"integrity": integ, "bench": {"cells": cells}, "serving": {"effect_pct": sv}}
    ok(decide_086(r86())["decision"] == "default_on_VKQ_REG_auto", "086v2 default-on")
    ok(decide_086(r86(long_eff=1.015))["decision"] == "fold_opt_in_default_off", "086v2 inside 2% floor -> opt-in")
    ok(decide_086(r86(long_eff=1.06, idc=1.07))["decision"] == "fold_opt_in_default_off", "086v2 identity control 7% raises floor -> opt-in")
    ok(decide_086(r86(short_eff=0.96))["decision"] == "fold_opt_in_default_off", "086v2 d0 regression -> opt-in")
    ok(decide_086(r86(sv=-3.0))["decision"] == "fold_opt_in_default_off", "086v2 serving regression -> opt-in")
    ok(decide_086(r86(integ=ok_i | {"fa_unset_bit_identical": None}))["decision"] == "hold", "086v2 integrity unmeasured -> hold")
    ok(decide_086(r86(integ=ok_i | {"tbo_fa_100": False}))["decision"] == "hold", "086v2 tbo fail -> hold")
    rr = r86(); rr["bench"]["cells"]["tg32@d32768"]["Bauto"] = [106.0]
    ok(decide_086(rr)["decision"] == "hold", "086v2 cell n<3 -> hold")
    rs = r86(); rs["serving"] = {"skipped": "deadline"}
    ok(decide_086(rs)["decision"] == "fold_opt_in_default_off", "086v2 serving unmeasured -> not default-on")
    print(f"selftest OK ({n} checks)"); return 0

def pin():
    lines = []
    for it in ("052", "050", "051", "086"):
        f = PRE_DIR / it / "PREDECLARED.md"
        if f.exists(): lines.append(f"{sha256_file(f)}  {it}/PREDECLARED.md")
    f = PRE_DIR / PRE086[0] / PRE086[1]
    if f.exists(): lines.append(f"{sha256_file(f)}  {PRE086[0]}/{PRE086[1]}")
    (PRE_DIR / "PINNED.sha256").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); return 0

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("item", nargs="?", choices=list(RUNNERS))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--execute", action="store_true")
    g.add_argument("--decide"); g.add_argument("--selftest", action="store_true"); g.add_argument("--pin", action="store_true")
    g.add_argument("--check-builds", action="store_true", help="precondition probe for filler_run2.sh --precond-cmd: rc 0 builds+knobs present, rc 5 build-missing (message on stdout)")
    ap.add_argument("--run-dir")
    a = ap.parse_args(argv)
    if a.selftest: return selftest()
    if a.pin: return pin()
    if not a.item: ap.error("item required")
    global STANDIN
    STANDIN = bool(a.dry_run and os.environ.get("V11_DRY_ARM_STANDIN") == "1")
    try:
        if a.check_builds:
            c = Ctx(a.item, RUNS / "_check", False); c.res = {}
            if a.item == "086":
                B = arm_build("086"); libs = [l for l in sorted((B / "bin").glob("libggml-cpu.so*")) if l.is_file() and not l.is_symlink()]
                if not any(strings_has(l, "GGML_FA_VKQ_REG") for l in libs): die(5, f"build-missing: GGML_FA_VKQ_REG not in {B}")
                if not (CORR086 / "INTEGRITY.json").exists(): die(5, f"build-missing: {CORR086}/INTEGRITY.json (correctness half pending)")
            else:
                B = arm_build(a.item)
                if a.item == "051":
                    libs = [l for l in sorted((B / "bin").glob("libggml-hip.so*")) if l.is_file() and not l.is_symlink()]
                    for knob in ("GGML_CUDA_FA_CDNA_D256_MAX_COLS", "GGML_CUDA_FA_CDNA_D256_TUNE"):
                        if not any(strings_has(l, knob) for l in libs): die(5, f"build-missing: {knob} not in {B}")
            require_c("cpu" if a.item == "086" else "gpu"); print(f"builds ready: {B}"); return 0
        if a.decide:
            res = json.loads(Path(a.decide).read_text()); print(json.dumps(DECIDE[a.item](res), indent=1, default=str)); return 0
        dry = not a.execute
        rd = Path(a.run_dir) if a.run_dir else RUNS / ("086v2" if a.item == "086" else a.item) / (("dry-" if dry else "") + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        sha = check_predeclared(a.item, rd)
        ctx = Ctx(a.item, rd, a.execute); ctx.res["predeclared_sha256"] = sha
        ctx.res["manifest"] = {"item": a.item, "started": now_iso(), "mode": "dry-run" if dry else "execute", "driver_sha256": sha256_file(__file__), "filler_id": os.environ.get("FILLER_ID")}
        if a.execute:
            inside_claim_or_die()
            if RUNNERS[a.item] is not run_086:
                bad = gpu_preflight(ctx)
                if bad: die(6, f"REFUSE preflight: {bad}")
            else:
                bad = cpu_preflight(ctx)
                if bad: die(6, f"REFUSE preflight: {bad}")
        RUNNERS[a.item](ctx, dry)
        (rd / "manifest.json").write_text(json.dumps(ctx.res["manifest"], indent=1))
        if a.execute: write_decision(ctx)
        else: ctx.say("dry-run complete; nothing launched")
        return 0
    except Refuse as e:
        print(f"{e}"); return e.rc

if __name__ == "__main__":
    sys.exit(main())
