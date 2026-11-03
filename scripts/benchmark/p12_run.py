#!/usr/bin/env python3
"""P1-2 / INF-22: corrected-placement re-measurement of E5 Stage-B (batched decode), a7-placement1 manifests.
Per cell: kernel guard -> sudo drop_caches -> numa_evict -> ONE harness invocation (server_numa_np_sweep.py, unchanged,
one manifest) with an in-window placement sampler (numa_placement_check.sh) on every llama-server in our own tree.
  p12_run.py --dry-run     model-free: guard, manifests, harness dry-run (no --execute), linkage-free; writes smoke RESULT.json
  p12_run.py               real run (needs FILLER_DEADLINE/FILLER_STOPFILE from the row engine, or env-free for manual use)
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, time

BASE = "/mnt/raid0/llm/tmp/p12-e5-stageb-20261103"
OUT = BASE + "/out"
RES = "/mnt/raid0/llm/epyc-inference-research"
PY = "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python"
HARN = RES + "/scripts/benchmark/server_numa_np_sweep.py"
MAN = RES + "/data/batched_decode/e5_manifests_a7_placement1"
EVICT = RES + "/scripts/utils/numa_evict.py"
PCHECK = RES + "/scripts/utils/numa_placement_check.sh"
STORE = "/mnt/raid0/llm/kernels/builds/cpu-20261103-181483cfc-lbt"
SERVER = STORE + "/bin/llama-server"
COEXIST = r"kernels/builds/gpu-|build-hip"          # YARN's champion GPU server (port 18437); GPU is never touched
# (model_key, group, [(cell, role)]) in run order. role: grid | parity | replicate
Q = "qwen36_q8_0"; N = "qwen3_next_80b"
def ladder(mk, cfg, shape, nps): return [f"{mk}-{cfg}-np{n}-{shape}" for n in nps]
GRID = [
 (Q, ladder(Q, "C1", "a7full", [1, 2, 4, 8, 16]) + [f"{Q}-C1-np1-e1parity-a7full"]),
 (Q, ladder(Q, "C1b", "a7half", [1, 2, 4, 8, 16])),
 (N, ladder(N, "C1", "a7full", [1, 2, 4, 8])),
 (N, ladder(N, "C1b", "a7half", [1, 2, 4, 8])),
]
REPLICATE = [f"{Q}-C1-np8-a7full", f"{N}-C1-np8-a7full"]   # noise estimate, run last, only if time remains
FLAT = 1.03   # futility: aggregate at 2k streams < 1.03x aggregate at k streams -> skip the higher np of that ladder

def guard():
    """Refuse anything that is not the champion-candidate store build (operator directive 2026-11-03)."""
    real = os.path.realpath(SERVER)
    bad = re.search(r"kernels/production|ffc1bac82|/llama\.cpp/build|/llama\.cpp-?[^/]*/build|production-consolidated", real + " " + SERVER)
    if bad or not real.startswith("/mnt/raid0/llm/kernels/builds/cpu-20261103-181483cfc-lbt/bin/"):
        sys.exit(f"REFUSED: llama-server {real} is not the champion-candidate store build (production/frozen path)")
    sums = {l.split()[1]: l.split()[0] for l in open(STORE + "/SHA256SUMS") if len(l.split()) == 2}
    h = hashlib.sha256(open(real, "rb").read()).hexdigest()
    if sums.get("bin/llama-server") != h:
        sys.exit("REFUSED: llama-server sha256 differs from the store SHA256SUMS")
    ident = json.load(open(STORE + "/BUILD-IDENTITY.json"))
    if not ident.get("source_sha", "").startswith("181483cfc"):
        sys.exit("REFUSED: BUILD-IDENTITY source_sha is not 181483cfc")
    return real

def manifest(cell):
    p = glob.glob(f"{MAN}/*/{cell}.json")
    if len(p) != 1: sys.exit(f"REFUSED: manifest for {cell} not found uniquely")
    return p[0]

def children(pid):
    out = []
    for f in glob.glob(f"/proc/{pid}/task/*/children"):
        try: out += [int(x) for x in open(f).read().split()]
        except OSError: pass
    return out

def tree(root):
    q, seen = [root], []
    while q:
        c = q.pop(); seen.append(c); q += children(c)
    return seen

def rss_gib(pid):
    try: return int(open(f"/proc/{pid}/statm").read().split()[1]) * 4096 / 2**30
    except (OSError, ValueError, IndexError): return 0.0

def comm(pid):
    try: return open(f"/proc/{pid}/comm").read().strip()
    except OSError: return ""

LOADED_FRAC = 0.90      # sampling starts only once server RSS >= 90% of the model file (cold mmap load is done)
SAMPLE_GAP_S = 45       # two samples this far apart (two-sample persistence)
POLL_S = 5
_now = time.time        # injectable clock (tests)

def sample(proc, thr, logdir, state, model_gib=None):
    """One pass of the in-window sampler over OUR descendants (never a name scan): comm is a check on pids from the walk.
    Sampling starts after LOAD (RSS >= LOADED_FRAC * model file size; without a size, RSS stable for 2 polls), then takes
    up to 2 samples SAMPLE_GAP_S apart. Earlier flaw: required stable RSS after mapping, so 78-102 s cold-load cells got none."""
    for pid in tree(proc.pid):
        if comm(pid) != "llama-server" or rss_gib(pid) < 1.0: continue
        s = state.setdefault(pid, {"prev": 0.0, "stable": 0, "taken": 0, "last": 0.0, "loaded": False})
        r = rss_gib(pid)
        s["stable"] = s["stable"] + 1 if r <= s["prev"] + 0.01 else 0
        s["prev"] = r
        if not s["loaded"]:
            s["loaded"] = r >= LOADED_FRAC * model_gib if model_gib else s["stable"] >= 2
        if s["loaded"] and s["taken"] < 2 and _now() - s["last"] >= SAMPLE_GAP_S:
            rc = subprocess.run(["bash", PCHECK, str(pid), "--threshold", str(thr), "--label", f"p12:{os.path.basename(logdir)}:pid{pid}"],
                                capture_output=True, text=True)
            s["taken"] += 1; s["last"] = _now()
            with open(f"{logdir}/placement.pid{pid}.s{s['taken']}.log", "w") as f: f.write(rc.stdout + rc.stderr + f"\nrc={rc.returncode}\n")
            s.setdefault("rcs", []).append(rc.returncode)

def placement_failure(state, wall_s):
    """LOUD failure text when the server exited with no placement sample taken; None when at least one sample exists."""
    if any(v.get("taken") for v in state.values()): return None
    return f"placement not sampled: server exited at T={round(wall_s)}s"

def cell_ok(rundir, cell):
    try: rows = [json.loads(l) for l in open(rundir + "/cells.jsonl")]
    except OSError: return None
    return next((r for r in rows if r.get("cell_id") == cell), None)

def run_cell(cell, dry, deadline, stopfile):
    man = manifest(cell)
    m = json.load(open(man))
    thr = 40 if m["instances"][0]["numactl_policy"].startswith("interleave") else 60   # 4-node interleave vs 2-node half
    rid = f"p12-{cell}"
    logdir = f"{OUT}/{cell}"; os.makedirs(logdir, exist_ok=True)
    cmd = [PY, "-I", HARN, "--cell-manifest", man, "--run-id", rid, "--output-root", OUT + "/runs",
           "--llama-server", SERVER, "--coexist-allow-pattern", COEXIST]
    if not dry:
        cmd += ["--execute", "--i-have-operator-grant"] + (["--allow-host-health-warning"] if os.environ.get("P12_INDICATIVE") == "1" else [])   # the grant IS the operator-approved row launched via run_direct.sh
        for step in (["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                     ["python3", "-I", EVICT, "--target-gib", "40"]):
            rc = subprocess.run(step, capture_output=True, text=True)
            open(logdir + "/pre-cell.log", "a").write(f"$ {' '.join(step)}\nrc={rc.returncode}\n{rc.stdout}{rc.stderr}\n")
            if step[0] == "sudo" and rc.returncode: return {"cell": cell, "status": "DROP_CACHES_FAILED"}
    try: model_gib = os.path.getsize(m["model_path"]) / 2**30
    except (OSError, KeyError): model_gib = None
    t0 = time.time()
    with open(logdir + "/harness.log", "w") as lg:
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, start_new_session=False)
        state = {}
        while p.poll() is None:
            if not dry: sample(p, thr, logdir, state, model_gib)
            if deadline and time.time() > deadline: p.terminate(); break
            time.sleep(POLL_S)
    res = {"cell": cell, "harness_rc": p.returncode, "wall_s": round(time.time() - t0), "placement_threshold": thr,
           "placement_rcs": {str(k): v.get("rcs") for k, v in state.items()}}
    if not dry:
        pf = placement_failure(state, time.time() - t0)
        if pf:
            res["placement_failure"] = pf
            print("LOUD FAILURE [" + cell + "] " + pf, file=sys.stderr, flush=True)
            open(logdir + "/PLACEMENT_NOT_SAMPLED.txt", "w").write(pf + "\n")
        row = cell_ok(f"{OUT}/runs/{rid}", cell)
        agg = (row or {}).get("aggregate_decode_tps")
        pl_ok = bool(state) and all(v.get("rcs") and all(r == 0 for r in v["rcs"]) for v in state.values())
        res.update(aggregate_decode_tps=agg, decision_grade_harness=(row or {}).get("decision_grade"),
                   placement_proof="PASS" if pl_ok else ("FAIL_NOT_SAMPLED" if pf else "FAIL_OR_MISSING"),
                   status="OK" if (row and not row.get("cell_error") and pl_ok) else "NOT_DECISION_GRADE")
    else:
        res["status"] = "DRYRUN_OK" if p.returncode == 0 else "DRYRUN_FAIL"
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--only")
    a = ap.parse_args()
    dry = a.dry_run or os.environ.get("FILLER_DRYRUN") == "1" or os.environ.get("FILLER_SMOKE") == "1"
    guard()
    os.makedirs(OUT, exist_ok=True)
    stop = os.environ.get("FILLER_STOPFILE", BASE + "/STOP")
    dl = float(os.environ.get("FILLER_DEADLINE", "0")) or None
    cells = [c for _, grp in GRID for c in grp]
    if dry or a.only:
        cells = [a.only] if a.only else [GRID[0][1][0]]            # smoke = ONE cell's dry-run (no --execute, no spawn)
    results, agg = [], {}
    res_path = OUT + "/RESULT.json"
    if os.path.exists(res_path): os.remove(res_path)
    def flush(status):
        json.dump({"schema": "p12.e5stageb.result.v1", "status": status, "kernel_store": STORE, "cells": results,
                   "planned": len(cells), "note": "descriptive; formal claims need MEASUREMENT.md grammar"}, open(res_path + ".tmp", "w"), indent=1)
        os.replace(res_path + ".tmp", res_path)
    for c in cells:
        if os.path.exists(stop): break
        grp = next(g for _, g in GRID if c in g)
        m = re.search(r"-np(\d+)-", c); n = int(m.group(1)) if m else 0
        # futility: skip np=2k when 2k/k aggregate gain on this ladder was < FLAT at the previous step
        prev = agg.get((id(grp), n // 2)); prev2 = agg.get((id(grp), n // 4)) if n >= 4 else None
        if not dry and prev and prev2 and prev < FLAT * prev2:
            results.append({"cell": c, "status": "SKIPPED_FUTILITY", "reason": f"np{n//2} < {FLAT}x np{n//4}"}); flush("RUNNING"); continue
        r = run_cell(c, dry, dl, stop); results.append(r)
        if r.get("aggregate_decode_tps") and n: agg[(id(grp), n)] = r["aggregate_decode_tps"]
        flush("RUNNING")
        if dl and time.time() > dl: break
    if not dry and not os.path.exists(stop) and not (dl and time.time() > dl):
        for c in REPLICATE:                                          # noise replicate, last
            if dl and time.time() + 900 > dl: break
            results.append(dict(run_cell(c, False, dl, stop), role="replicate")); flush("RUNNING")
    flush("SMOKE_DRYRUN" if dry else "COMPLETE")
    return 0 if all(r.get("status") in ("OK", "DRYRUN_OK", "SKIPPED_FUTILITY", "NOT_DECISION_GRADE") for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())
