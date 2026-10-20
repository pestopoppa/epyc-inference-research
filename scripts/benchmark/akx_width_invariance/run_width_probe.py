#!/usr/bin/env python3
"""AKX-ALL-26 width-invariance probe driver (stdlib only).

Runs the out-of-tree `width_probe` binary once PER PREDICTED ID (a ggml assert can only kill one id),
with an explicit, minimal environment per predicted-row backend variant, then writes

  <out>/cells.jsonl          every record emitted by the probe (run / k / cell / row_done), id order
  <out>/first_fail_table.md  first-fail k per id/sub-shape (op, type, backend, shape, route text)
  <out>/scorecard.json       frozen predictions joined with observations: confirmed / partially /
                             falsified (+ unsupported / error / voided / not_probed / missing),
                             the A/A (repeat) summary and the PC-1 positive-control result.

Sub-commands
  run    --backend cpu|rocm0 --build-bin DIR --cases predicted.jsonl --out DIR [options]
  score  --cases predicted.jsonl --out DIR --inputs DIR [DIR ...]   (merge CPU + HIP halves)

Resumption: per-id part files under <out>/parts/; an id whose part file ends in a `row_done` record is
skipped on re-run (use --rerun-errors to retry ids that ended in a driver-level ERR).
Sharding: --ids a,b,c | --ids-file FILE | --shard I/N (stable modulo over the selected ids).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINKAGE_SCRIPT = "/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
DEFAULT_TREE = "/mnt/raid0/llm/llama.cpp-experimental-champref-4348de400-20261006"


def load_rows(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def family_of(row_backend):
    if row_backend == "rocm0":
        return "rocm0"
    if row_backend.startswith("cpu") or row_backend == "graph":
        return "cpu"
    return None


def env_for(row_backend, run_backend, build_bin):
    """Minimal env: drop every inherited GGML_* knob and LD_LIBRARY_PATH, then set exactly what the row says."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_") and k != "LD_LIBRARY_PATH"}
    env["LD_LIBRARY_PATH"] = build_bin if run_backend == "cpu" else build_bin + ":/opt/rocm/lib"
    if run_backend == "cpu":
        if row_backend in ("cpu", "graph", "PC-1"):
            env["GGML_IQK"] = "1"                       # production CPU env (prediction assumption 3)
        elif row_backend == "cpu-native":
            pass                                        # GGML_IQK unset
        else:
            m = re.fullmatch(r"cpu\[(.*)\]", row_backend)
            if not m:
                raise SystemExit("unknown cpu row backend %r" % row_backend)
            for kv in m.group(1).split(","):
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def readelf_runpath(binary):
    try:
        out = subprocess.run(["readelf", "-d", binary], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return None, []
    rp = re.findall(r"\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[(.*)\]", out)
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.*)\]", out)
    return (rp[0] if rp else None), needed


def part_state(path):
    """returns 'done', 'driver_err', or None (absent/incomplete)."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("record") == "row_done":
                last = "driver_err" if r.get("driver_status") == "ERR" else "done"
    return last


def safe_name(i):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", i)


def cmd_run(a):
    build_bin = os.path.abspath(a.build_bin)
    binary = os.path.abspath(a.binary) if a.binary else None
    if not binary:
        raise SystemExit("--binary is required (width_probe_cpu for --backend cpu, width_probe_hip for --backend rocm0)")
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "parts"), exist_ok=True)
    rows = load_rows(a.cases)
    sel = [r for r in rows if family_of(r["backend"]) == a.backend]
    if a.backend == "rocm0":
        sel = [r for r in sel if r["backend"] == "rocm0"]
    ids = [r["id"] for r in sel]
    want = None
    if a.ids:
        want = set(x for x in a.ids.split(",") if x)
    if a.ids_file:
        with open(a.ids_file) as f:
            want = (want or set()) | set(x.strip() for x in f if x.strip())
    if want is not None:
        unknown = sorted(want - set(ids) - {"PC-1"})
        if unknown:
            print("[driver] WARNING ids not in this backend half (ignored): %s" % ", ".join(unknown), file=sys.stderr)
        ids = [i for i in ids if i in want]
    run_pc1 = not a.no_pc1 and (want is None or "PC-1" in want)
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        ids = [i for n, i in enumerate(ids) if n % sn == si]
        run_pc1 = run_pc1 and si == 0
    if run_pc1:
        ids = ["PC-1"] + ids
    by_id = {r["id"]: r for r in rows}

    # ---- linkage / binding guard (file inspection only; nothing is executed here)
    rp, needed = readelf_runpath(binary)
    print("[driver] binary=%s RUNPATH=%s NEEDED=%s" % (binary, rp, ",".join(n for n in needed if "ggml" in n)), file=sys.stderr)
    if rp is not None and build_bin not in rp.split(":"):
        raise SystemExit("[driver] FATAL: binary RUNPATH %r does not contain --build-bin %s (would load another tree's ggml)" % (rp, build_bin))
    for lib in ("libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0") + (("libggml-hip.so.0",) if a.backend == "rocm0" else ()):
        if not os.path.exists(os.path.join(build_bin, lib)):
            raise SystemExit("[driver] FATAL: %s missing from --build-bin %s" % (lib, build_bin))
    tree = a.tree or DEFAULT_TREE
    print("[driver] expected linkage check (run it before trusting results):\n  %s %s %s"
          % (LINKAGE_SCRIPT, binary, build_bin), file=sys.stderr)
    print("[driver] (tree for headers: %s)" % tree, file=sys.stderr)
    if a.verify_linkage:
        rc = subprocess.run([LINKAGE_SCRIPT, binary, build_bin], env=env_for("cpu" if a.backend == "cpu" else "rocm0", a.backend, build_bin)).returncode
        if rc != 0:
            raise SystemExit("[driver] FATAL: verify_ggml_linkage.sh rc=%d" % rc)

    manifest = {
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(), "backend": a.backend, "build_bin": build_bin,
        "binary": binary, "cases": os.path.abspath(a.cases), "threads": a.threads, "kmax": a.kmax, "repeats": a.repeats,
        "seed": a.seed, "ids": ids, "shard": a.shard, "argv": sys.argv,
    }
    with open(os.path.join(out, "run_manifest.%s.json" % (a.shard.replace("/", "of") if a.shard else "all")), "w") as f:
        json.dump(manifest, f, indent=1)

    worst = 0
    for n, i in enumerate(ids):
        part = os.path.join(out, "parts", safe_name(i) + ".jsonl")
        st = part_state(part)
        if st == "done" or (st == "driver_err" and not a.rerun_errors):
            print("[driver] (%d/%d) %s: resume-skip (%s)" % (n + 1, len(ids), i, st), file=sys.stderr)
            continue
        if os.path.exists(part):
            os.remove(part)                                  # incomplete: start the id over
        row_backend = "PC-1" if i == "PC-1" else by_id[i]["backend"]
        env = env_for(row_backend, a.backend, build_bin)
        cmd = [binary, "--backend", a.backend, "--cases", os.path.abspath(a.cases), "--ids", i, "--kmax", str(a.kmax),
               "--repeats", str(a.repeats), "--threads", str(a.threads), "--seed", str(a.seed), "--out", part]
        if a.quant_threads:
            cmd += ["--quant-threads", str(a.quant_threads)]
        envshow = " ".join("%s=%s" % (k, env[k]) for k in sorted(env) if k.startswith("GGML_") or k == "LD_LIBRARY_PATH")
        print("[driver] (%d/%d) %s [%s] %s" % (n + 1, len(ids), i, envshow, " ".join(cmd)), file=sys.stderr)
        if a.dry_run:
            continue
        with open(os.path.join(out, "parts", safe_name(i) + ".stderr.log"), "w") as errf:
            try:
                rc = subprocess.run(cmd, env=env, stdout=errf, stderr=subprocess.STDOUT, timeout=a.timeout_per_id).returncode
                why = "exit %d" % rc
            except subprocess.TimeoutExpired:
                rc, why = 124, "timeout after %ss (child killed by subprocess.run)" % a.timeout_per_id
        if rc == 4 and i == ids[0] and part_state(part) is None:
            raise SystemExit("[driver] FATAL: probe exited 4 (backend binding / usage error) on the first id; see %s" % part.replace(".jsonl", ".stderr.log"))
        if part_state(part) != "done":
            tail = ""
            try:
                with open(os.path.join(out, "parts", safe_name(i) + ".stderr.log")) as f:
                    tail = f.read()[-1500:]
            except OSError:
                pass
            with open(part, "a") as f:
                f.write(json.dumps({"record": "cell", "id": i, "sub": -1, "sub_label": "(process)", "backend_row": row_backend, "status": "ERR",
                                    "verdict": "ERR", "first_fail_k": None, "reason": "probe process did not complete: %s; stderr tail: %s" % (why, tail)}) + "\n")
                f.write(json.dumps({"record": "row_done", "id": i, "backend_row": row_backend, "driver_status": "ERR"}) + "\n")
        worst = max(worst, rc if rc in (2, 3) else (1 if rc not in (0,) else 0))
    if a.dry_run:
        return 0
    write_outputs(a.cases, out, [out])
    return worst


# ------------------------------------------------------------------------------------------------
def collect(dirs):
    recs = []
    for d in dirs:
        pdir = os.path.join(d, "parts")
        if not os.path.isdir(pdir):
            continue
        for fn in sorted(os.listdir(pdir)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(pdir, fn)) as f:
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            try:
                                recs.append(json.loads(ln))
                            except json.JSONDecodeError:
                                pass
    return recs


def observe(cells):
    """Fold the sub-shape cell records of one id into one observation."""
    if not cells:
        return {"verdict": "MISSING", "first_fail_k": None}
    st = [c.get("status") for c in cells]
    subs = [{"sub": c.get("sub"), "sub_label": c.get("sub_label"), "status": c.get("status"), "first_fail_k": c.get("first_fail_k"),
             "reason": c.get("reason")} for c in cells]
    ok = [c for c in cells if c.get("status") == "OK"]
    if any(s == "UNSTABLE" for s in st):
        v, k = "UNSTABLE", None
    elif ok:
        ks = [c["first_fail_k"] for c in ok if c.get("first_fail_k")]
        v, k = ("FAIL", min(ks)) if ks else ("PASS", None)
    elif all(s == "UNSUPPORTED" for s in st):
        v, k = "UNSUPPORTED", None
    elif all(s == "NOT_PROBED" for s in st):
        v, k = "NOT_PROBED", None
    else:
        v, k = "ERR", None
    okks = sorted(set(c.get("first_fail_k") for c in ok), key=lambda x: (x is None, x or 0))
    return {"verdict": v, "first_fail_k": k, "subs": subs,
            "shape_dependent": len(okks) > 1,
            "coverage": "partial" if ok and len(ok) < len(cells) else ("full" if ok else "none")}


def mark(pred_v, pred_k, obs):
    ov, ok_ = obs["verdict"], obs["first_fail_k"]
    if ov == "MISSING":
        return "missing", "no result for this id"
    if ov == "ERR":
        return "error", "probe error; not scorable until re-run"
    if ov == "UNSTABLE":
        return "voided", "a repeat was unstable (run-to-run nondeterminism); width verdict not attributable"
    if ov == "NOT_PROBED":
        return "not_probed", "graph-level prediction; the op-level probe does not exercise it"
    if pred_v == "UNSUPPORTED":
        return ("confirmed", "UNSUPPORTED as predicted") if ov == "UNSUPPORTED" else ("falsified", "predicted UNSUPPORTED but the cell was constructed and computed (%s)" % ov)
    if ov == "UNSUPPORTED":
        return "falsified", "predicted %s but observed UNSUPPORTED (could not be constructed / backend refused)" % pred_v
    if pred_v != ov:
        return "falsified", "verdict differs: predicted %s@%s observed %s@%s" % (pred_v, pred_k, ov, ok_)
    if pred_v == "PASS" or pred_k == ok_:
        return "confirmed", "verdict and first-fail k equal"
    return "partially", "same verdict, first-fail k differs: predicted %s observed %s" % (pred_k, ok_)


def write_outputs(cases, out, dirs):
    rows = load_rows(cases)
    recs = collect(dirs)
    os.makedirs(out, exist_ok=True)
    if dirs == [out]:
        with open(os.path.join(out, "cells.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    cells_by = {}
    for r in recs:
        if r.get("record") == "cell":
            key = (r["id"], r.get("backend_row") if r["id"] != "PC-1" else r.get("backend_bound", r.get("backend_row")))
            cells_by.setdefault(key, []).append(r)
    present_backends = set(k[1] for k in cells_by)
    run_recs = [r for r in recs if r.get("record") == "run"]
    bound = sorted(set((r.get("backend_bound"), r.get("device_description")) for r in run_recs))
    probed_families = set()
    for r in run_recs:
        probed_families.add("rocm0" if (r.get("backend_requested") == "rocm0") else "cpu")

    score_rows = []
    counts = {}
    for p in rows:
        fam = family_of(p["backend"])
        cells = cells_by.get((p["id"], p["backend"]), [])
        if fam not in probed_families and not cells:
            m, why, obs = "not_in_scope", "backend half %s not part of these inputs" % fam, {"verdict": None, "first_fail_k": None}
        else:
            obs = observe(cells)
            m, why = mark(p["predicted_verdict"], p["predicted_first_fail_k"], obs)
        counts[m] = counts.get(m, 0) + 1
        score_rows.append({"id": p["id"], "backend": p["backend"], "op": p["op"], "type": p["type"], "shape": p["shape"],
                           "predicted_verdict": p["predicted_verdict"], "predicted_first_fail_k": p["predicted_first_fail_k"],
                           "confidence": p.get("confidence"), "inference_flag": p.get("inference_flag"),
                           "observed": obs, "mark": m, "mark_reason": why, "route": p.get("route"), "source_loc": p.get("source_loc")})

    krecs = [r for r in recs if r.get("record") == "k"]
    unstable = [{"id": r["id"], "sub_label": r.get("sub_label"), "k": r["k"], "backend": r.get("backend_bound")} for r in krecs if r.get("status") == "UNSTABLE"]
    k1 = [{"id": r["id"], "sub_label": r.get("sub_label"), "backend": r.get("backend_bound")} for r in krecs if r.get("k") == 1 and r.get("width_differs")]
    pc1 = []
    for (i, b), cs in sorted(cells_by.items()):
        if i == "PC-1":
            for c in cs:
                pc1.append({"backend": b, "status": c.get("status"), "verdict": c.get("verdict"), "first_fail_k": c.get("first_fail_k"),
                            "expected_verdict": "FAIL", "control_ok": c.get("verdict") == "FAIL"})
    aa = {"k_records": len(krecs), "k_records_ok": sum(1 for r in krecs if r.get("status") == "OK"),
          "unstable_k_records": unstable, "a_a_clean": len(unstable) == 0 and len(krecs) > 0,
          "k1_identity_violations": k1}
    sc = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(), "inputs": [os.path.abspath(d) for d in dirs],
          "cases": os.path.abspath(cases), "bound_backends": [{"name": n, "description": d} for n, d in bound],
          "mark_counts": counts, "a_a": aa, "pc1": pc1,
          "pc1_ok_all": bool(pc1) and all(x["control_ok"] for x in pc1),
          "marking_rules": {
              "confirmed": "predicted verdict AND predicted first-fail k equal the observation (PASS/UNSUPPORTED: verdict equal)",
              "partially": "same FAIL verdict, different first-fail k",
              "falsified": "verdict differs (incl. predicted UNSUPPORTED but computed, or predicted PASS/FAIL but observed UNSUPPORTED)",
              "voided": "an A/A repeat was unstable for this id", "error": "probe ERR (re-run)", "not_probed": "graph-level row",
              "missing": "no record", "not_in_scope": "the other backend half is not in these inputs",
              "multi-shape ids": "observed first-fail k = min over computed sub-shapes; per-sub results in observed.subs; shape_dependent flags disagreement"},
          "rows": score_rows}
    with open(os.path.join(out, "scorecard.json"), "w") as f:
        json.dump(sc, f, indent=1)

    # ---- first-fail table
    L = ["# AKX-ALL-26 width-invariance first-fail table", "",
         "Inputs: %s  " % ", ".join(os.path.abspath(d) for d in dirs),
         "Bound backends: %s  " % "; ".join("%s (%s)" % b for b in bound),
         "A/A clean: **%s** (%d k-records, %d unstable); k=1 identity violations: %d  " % (aa["a_a_clean"], aa["k_records"], len(unstable), len(k1)),
         "PC-1 (must FAIL): %s  " % ("; ".join("%s: %s@%s" % (x["backend"], x["verdict"], x["first_fail_k"]) for x in pc1) or "not run"),
         "Marks: %s" % ", ".join("%s=%d" % kv for kv in sorted(counts.items())), "",
         "first-fail k = smallest width k in 1..kmax whose batched output is not byte-identical to the k serial single-column ops; "
         "`-` = bit-equal for every k. One row per id and sub-shape (multi-shape ids: the id verdict is the min over sub-shapes).", "",
         "| id | backend | op | type | sub-shape | status | first-fail k | predicted | mark | route (frozen prediction, truncated) |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def cell_txt(s):
        return str(s).replace("|", "\\|").replace("\n", " ")
    for s in score_rows:
        if s["mark"] == "not_in_scope":
            continue
        pred = "%s@%s" % (s["predicted_verdict"], s["predicted_first_fail_k"]) if s["predicted_first_fail_k"] else s["predicted_verdict"]
        route = (s["route"] or "")[:220] + ("..." if s["route"] and len(s["route"]) > 220 else "")
        subs = s["observed"].get("subs") or [{"sub_label": "-", "status": s["observed"]["verdict"], "first_fail_k": None}]
        for sub in subs:
            ffk = sub.get("first_fail_k") if sub.get("first_fail_k") else ("-" if sub.get("status") == "OK" else "")
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % tuple(cell_txt(x) for x in (
                s["id"], s["backend"], s["op"], s["type"], sub.get("sub_label"), sub.get("status"), ffk, pred, s["mark"], route)))
    for x in pc1:
        L.append("| PC-1 | %s | MUL_MAT split-K control | F32 | W[4096,512] | %s | %s | FAIL (expected) | %s | positive control |"
                 % (x["backend"], x["status"], x["first_fail_k"], "control-ok" if x["control_ok"] else "CONTROL-BROKEN"))
    if unstable:
        L += ["", "## Unstable (A/A) k-records", ""] + ["- %s %s k=%s (%s)" % (u["id"], u["sub_label"], u["k"], u["backend"]) for u in unstable]
    with open(os.path.join(out, "first_fail_table.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("[driver] wrote %s/{cells.jsonl,first_fail_table.md,scorecard.json}; marks=%s a_a_clean=%s pc1_ok=%s"
          % (out, counts, aa["a_a_clean"], sc["pc1_ok_all"]), file=sys.stderr)


def cmd_score(a):
    write_outputs(a.cases, os.path.abspath(a.out), [os.path.abspath(d) for d in a.inputs])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--backend", required=True, choices=["cpu", "rocm0"])
    r.add_argument("--build-bin", required=True, help="the llama.cpp build's bin/ holding libggml*.so (LD_LIBRARY_PATH is set to exactly this)")
    r.add_argument("--binary", help="compiled width_probe linked against --build-bin")
    r.add_argument("--tree", help="llama.cpp source tree whose ggml headers the binary was compiled against (reporting only)")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ids")
    r.add_argument("--ids-file")
    r.add_argument("--shard", help="I/N")
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--quant-threads", type=int)
    r.add_argument("--kmax", type=int, default=9)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--seed", default="674854")
    r.add_argument("--timeout-per-id", type=int, default=3600)
    r.add_argument("--no-pc1", action="store_true")
    r.add_argument("--rerun-errors", action="store_true")
    r.add_argument("--verify-linkage", action="store_true", help="also run verify_ggml_linkage.sh (ldd) before probing")
    r.add_argument("--dry-run", action="store_true")
    s = sp.add_parser("score")
    s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--inputs", nargs="+", required=True)
    a = ap.parse_args()
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_score(a))


if __name__ == "__main__":
    main()
