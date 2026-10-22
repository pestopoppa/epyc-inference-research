#!/usr/bin/env python3
"""KVU-12.2 MTP-on context-bank gates and bank cells (workspace-f8, 2026-10-22).

Task text (kv-unified-stack-rollout.md KVU-12.2): port upstream #30574 (draft state in the slot file), run gate (a) and gate (b) on that build against
the unported champion as control with the study's predeclared rule (restored acceptance >= 0.9x live AND restored decode >= 0.95x live, both layouts,
both suffixes); if it passes, run the MTP-on bank cells on the champion+#30574 binary, labelled with that binary. Banks are rebuilt per binary.

What this adds (the study driver `kvu_cpu_matrix.py` is invoked as a subprocess and NOT modified; `kvu_ckpt_arms.py` supplies the D4-calibrated
estimate model and the filler Budget/Calibrator):
  stage gates_control   kvu_cpu_matrix.py --gates --binary <unported champion image>          (the rule must FAIL here: the limitation exists)
  stage gates_cand      kvu_cpu_matrix.py --gates --binary <champion + #26004 + #30574 image>
  verdict               pure arithmetic on both gates.json (rule above) + server-log proof that the draft-state save/restore path RAN
  stage bank_*          kvu_cpu_matrix.py --build-bank (unified np4 bank, split np4 bank, split np2 bank) on the candidate        [only if the candidate passes]
  stage cells_*         kvu_cpu_matrix.py --from-bank, MTP-on, one launch per (layout, np) at L=8k                                    [only if the candidate passes]
Class: counts / byte-identity / acceptance = CORRECTNESS (build-role claim, any quiet-host state); decode tok/s is INDICATIVE and carries the binary label.
Filler contract: --stop-file between stages, --deadline-epoch (no new stage that cannot finish: est x calibrated factor + margin), --hard-end-epoch
(TERM the running driver, which kills its own server by captured PID), stages resume from state.json, end record RUN_RECORD.json.
Nothing here builds llama.cpp and nothing starts without the launcher's grant.
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kvu_cpu_matrix as M                                   # noqa: E402  (study driver, never edited here)
import kvu_ckpt_arms as A                                    # noqa: E402  (Calibrator, Budget, cold_prefill_s)

SCHEMA = "epyc.kvu122.bank_cells.v1"
KB = Path("/mnt/raid0/llm/kernels/builds")
CAND_DIR = KB / "cpu-20261020-dd3d9721c" / "bin"             # champion e33229348 + #26004 (1c999e15b) + #30574 (deb387b8b) + tests fix; build 10345
CTRL_DIR = KB / "cpu-20261006-4348de400" / "bin"             # latest CPU champion image BEFORE the slot-file ports (the unported control)
MARK_DRAFT = b"appended the draft state"                     # SLT_INF in save_slot_draft (deb387b8b); present only in builds carrying #30574
MARK_RESTORE = b"restored the draft state"
STUDY_V10_GATES = HERE.parent / "gates" / "gates.json"       # the 2026-10-20 v10 control record (decode_tps not captured then)
ACC_MIN, DEC_MIN = 0.9, 0.95                                 # predeclared (PLAN.md "MTP decision rule", coordinator 2026-10-20)
MODEL_KEY = "q36_35b"
BANK_CELLS = (("unified", 4), ("split", 4), ("unified", 2), ("split", 2))   # 4 MTP-on bank cells at L8k (see --lens for 32k at +~1.6 h)


# ----------------------------------------------------------------------------------------------- the predeclared rule (pure arithmetic)
def _acc(d):
    return (d["draft_acc"] / d["draft_n"]) if d and d.get("draft_n") else None


def gate_verdict(gates, acc_min=ACC_MIN, dec_min=DEC_MIN):
    """gates = list of run_gates() dicts (one per layout). PASS needs, for EVERY layout and suffix: output byte-identical (gate a), restored acceptance >=
    acc_min x live, restored decode tok/s >= dec_min x live (gate b), extend prompt_n == chunk (gate c). A missing decode_tps makes the rule
    not evaluable -> not a pass (the 2026-10-20 v10 record predates the field)."""
    rows, failed = [], []
    for g in gates:
        for s in ("A", "B"):
            live, rest = g.get(f"live{s}"), g.get(f"restored{s}")
            al, ar = _acc(live), _acc(rest)
            dl, dr = (live or {}).get("decode_tps"), (rest or {}).get("decode_tps")
            ar_ratio = (ar / al) if (al and ar is not None) else None
            dec_ratio = (dr / dl) if (dl and dr is not None) else None
            ident = g.get(f"identical{s}")
            err = bool((live or {}).get("err") or (rest or {}).get("err"))
            row = dict(mode=g.get("mode"), suffix=s, acc_live=al, acc_restored=ar, acc_ratio=ar_ratio, decode_live=dl, decode_restored=dr,
                       decode_ratio=dec_ratio, identical=ident, error=err)
            rows.append(row)
            tag = f"{g.get('mode')}/{s}"
            if err:
                failed.append(f"{tag}: request error")
            if ident is not True:
                failed.append(f"{tag}: gate (a) restored output differs from live")
            if ar_ratio is None or ar_ratio < acc_min:
                failed.append(f"{tag}: acceptance ratio {None if ar_ratio is None else round(ar_ratio, 3)} < {acc_min}")
            if dec_ratio is None:
                failed.append(f"{tag}: decode tok/s not captured (rule not evaluable)")
            elif dec_ratio < dec_min:
                failed.append(f"{tag}: decode ratio {round(dec_ratio, 3)} < {dec_min}")
        ext = g.get("extend") or {}
        if ext.get("ok") is not True:
            failed.append(f"{g.get('mode')}: gate (c) extend prompt_n {ext.get('prompt_n')} != {ext.get('expected_prompt_n')}")
    return dict(passed=(not failed and bool(rows)), rows=rows, failed=failed, rule=dict(acc_min=acc_min, decode_min=dec_min))


def count_markers(root):
    """how many server.log lines show the draft-state save/restore path RAN (proof the gates were not vacuous)"""
    n_save = n_restore = 0
    for p in Path(root).rglob("server.log"):
        try:
            t = p.read_bytes()
        except OSError:
            continue
        n_save += t.count(MARK_DRAFT)
        n_restore += t.count(MARK_RESTORE)
    return n_save, n_restore


def file_has(path, needle):
    with open(path, "rb") as f:
        while True:
            blk = f.read(1 << 24)
            if not blk:
                return False
            if needle in blk:
                return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- estimates (D4-calibrated model, per-key correction)
def decode_rate(threads):
    return A.MODELS[MODEL_KEY]["decode_tps"] * min(1.0, threads / 96.0 + 0.25)      # same scaling as est_gen_cell (ASSUMED below 96 threads)


def est_gates_s(threads, D=7680, chunk=1024, npred=256, load_s=A.LOAD_WARM_S[MODEL_KEY] + 60):
    pf = lambda n: A.cold_prefill_s(n, MODEL_KEY, threads)
    dec = npred / decode_rate(threads)
    per_mode = 2 * ((pf(D + 128) + dec) + pf(D) + 3.0 + (pf(128) + dec)) + 3.0 + pf(chunk) + 30.0
    return 2 * (per_mode + load_s)


def est_bank_s(nd, names, threads, load_s=A.LOAD_WARM_S[MODEL_KEY] + 60):
    pf = lambda n: A.cold_prefill_s(n, MODEL_KEY, threads)
    maxD = M.bank_tok(names[-1])
    steps = nd * (pf(maxD) + 30.0 * len(names))
    ctl = sum(pf(M.bank_tok(nm)) for nm in names)
    return load_s + steps + ctl + 180.0


def est_cell_launch_s(np_, names, waves, npred, threads, load_s=A.LOAD_WARM_S[MODEL_KEY] + 60):
    pf = lambda n: A.cold_prefill_s(n, MODEL_KEY, threads)
    t = load_s
    for nm in names:
        L = M.LS[nm]
        rest = np_ * (L * M.KV_BYTES_PER_TOKEN / 1.5e9 + 1.0)
        dec = npred / (decode_rate(threads) / max(1, np_) * 1.0) * (1 + 0.25 * L / 131072)
        t += waves * (rest + np_ * pf(200) / np_ + dec) + 60.0
    return t


# ----------------------------------------------------------------------------------------------- stage runner
def alive_ours(pid, port):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return False
    return f"--port {port}" in cmd


def reap_servers(out, port):
    """kill (by CAPTURED pid, from the driver's server.pid files, only if the cmdline still carries our port) and verify"""
    bad = []
    for pf in Path(out).rglob("server.pid"):
        try:
            pid = int(pf.read_text().strip())
        except (OSError, ValueError):
            continue
        if not alive_ours(pid, port):
            continue
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            t_end = time.time() + wait
            while time.time() < t_end and alive_ours(pid, port):
                time.sleep(1)
            if not alive_ours(pid, port):
                break
        if alive_ours(pid, port):
            bad.append(pid)
    return bad


class Stage:
    def __init__(self, name, argv, out, est_raw_s, kind):
        self.name, self.argv, self.out, self.est_raw_s, self.kind = name, argv, Path(out), est_raw_s, kind


def run_stage(st, a, state, cal, budget):
    prev = state.get(st.name)
    if prev and prev.get("rc") == 0:
        print(f"[{time.strftime('%FT%TZ', time.gmtime())}] skip {st.name} (done rc=0)", flush=True)
        return 0
    why = budget.stop_reason()
    if why:
        print(f"stop before {st.name}: {why}", flush=True)
        return 9
    planned = st.est_raw_s * cal.factor(MODEL_KEY, st.kind, "quarter")
    if not budget.fits(planned):
        print(f"{st.name}: planned {planned/60:.1f} min does not fit before the deadline (remaining {budget.remaining()}); not started", flush=True)
        return 9
    st.out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", "-I", str(HERE / "kvu_cpu_matrix.py")] + st.argv
    log = open(st.out / "stage.log", "ab")
    log.write((" ".join(cmd) + "\n").encode())
    t0 = time.time()
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] START {st.name} est_raw={st.est_raw_s/60:.1f} min planned={planned/60:.1f} min :: {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)       # own session; killed only by this captured pid
    rc = None
    while rc is None:
        rc = p.poll()
        if rc is not None:
            break
        if a.hard_end_epoch and time.time() >= a.hard_end_epoch:
            p.terminate()                                    # the driver's _cleanup kills its server by pid
            try:
                p.wait(45)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(15)
            rc = 124
            break
        time.sleep(5)
    wall = time.time() - t0
    bad = reap_servers(st.out, a.port)
    state[st.name] = dict(rc=rc, wall_s=round(wall, 1), est_raw_s=round(st.est_raw_s, 1), kind=st.kind, survivors=bad, ended=time.strftime("%FT%TZ", time.gmtime()))
    if rc == 0 and wall > 0:
        cal.observe(MODEL_KEY, st.kind, "quarter", st.est_raw_s, wall)
    save_state(a, state)
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] END {st.name} rc={rc} wall={wall/60:.1f} min (est_raw {st.est_raw_s/60:.1f})" + (f" SURVIVING SERVER PIDS {bad}" if bad else ""), flush=True)
    return rc


def save_state(a, state):
    tmp = Path(a.out) / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(Path(a.out) / "state.json")


def load_state(a):
    p = Path(a.out) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------------------------- plan
def common_argv(a, binary, out):
    return ["--out", str(out), "--binary", str(binary), "--cpu-list", a.cpu_list, "--threads", str(a.threads), "--port", str(a.port)]


def bank_names(a):
    return [x for x in a.lens.split(",")]


def build_stages(a, include_cells=True):
    out = Path(a.out)
    names = bank_names(a)
    cand, ctrl = CAND_DIR / "llama-server", CTRL_DIR / "llama-server"
    st = []
    gate_est = est_gates_s(a.threads)
    if not a.skip_control:
        st.append(Stage("gates_control", ["--gates"] + common_argv(a, ctrl, out / "gates_control") + ["--bank-dir", str(out / "bank_gates_control")],
                        out / "gates_control", gate_est, "gates"))
    st.append(Stage("gates_cand", ["--gates"] + common_argv(a, cand, out / "gates_cand") + ["--bank-dir", str(out / "bank_gates_cand")],
                    out / "gates_cand", gate_est, "gates"))
    if not include_cells:
        return st
    bd = out / "bank_cand"
    mf = out / "bank_manifest.json"
    layouts = []
    for lay, np_ in BANK_CELLS[: a.n_cells]:
        layouts.append((lay, np_))
    need_uni = any(l == "unified" for l, _ in layouts)
    max_uni_np = max([n for l, n in layouts if l == "unified"] or [0])
    if need_uni:                                          # unified bank = n_stream 1: valid for every np and pool; docs = largest np
        st.append(Stage("bank_unified", ["--build-bank", "--bank-kv", "unified", "--bank-docs", str(max_uni_np), "--bank-names", ",".join(names),
                                         "--bank-dir", str(bd), "--manifest", str(mf)] + common_argv(a, cand, out / "bank_unified"),
                        out / "bank_unified", est_bank_s(max_uni_np, names, a.threads), "bank"))
    for lay, np_ in sorted({(l, n) for l, n in layouts if l == "split"}, reverse=True):
        st.append(Stage(f"bank_split_np{np_}", ["--build-bank", "--bank-kv", "split", "--bank-np", str(np_), "--bank-names", ",".join(names),
                                                "--bank-dir", str(bd), "--manifest", str(mf)] + common_argv(a, cand, out / f"bank_split_np{np_}"),
                        out / f"bank_split_np{np_}", est_bank_s(np_, names, a.threads), "bank"))
    for lay, np_ in layouts:
        only = f"unified_np{np_}_pool524288" if lay == "unified" else f"split_np{np_}_slot65536"
        st.append(Stage(f"cells_{lay}_np{np_}", ["--from-bank", "--kv-modes", lay, "--nps", str(np_), "--lens", ",".join(names), "--pmodes", "independent",
                                                 "--pools", "524288", "--waves", str(a.waves), "--npredict", str(M.N_PREDICT_DEFAULT), "--only", only,
                                                 "--bank-dir", str(bd), "--manifest", str(mf)] + common_argv(a, cand, out / f"cells_{lay}_np{np_}"),
                        out / f"cells_{lay}_np{np_}", est_cell_launch_s(np_, names, a.waves, M.N_PREDICT_DEFAULT, a.threads), "cells"))
    return st


def cmd_plan(a):
    cal = A.Calibrator()
    tot = 0.0
    print(f"KVU-12.2 plan (threads={a.threads}, cpus={a.cpu_list}, lens={a.lens}, waves={a.waves}); estimates = D4-calibrated model x {A.SAFETY_UNSEEN} (unseen key)")
    print(f"  candidate {CAND_DIR / 'llama-server'}\n  control   {CTRL_DIR / 'llama-server'}")
    for s in build_stages(a):
        planned = s.est_raw_s * cal.factor(MODEL_KEY, s.kind, "quarter")
        tot += planned
        print(f"  {s.name:<22} raw {s.est_raw_s/60:6.1f} min  planned {planned/60:6.1f} min  :: kvu_cpu_matrix.py {' '.join(s.argv)}")
    g = sum(s.est_raw_s * A.SAFETY_UNSEEN for s in build_stages(a, include_cells=False))
    print(f"  GATES ONLY planned {g/60:.1f} min;  ALL planned {tot/60:.1f} min  (cells/banks run only if the candidate passes the rule)")
    return 0


# ----------------------------------------------------------------------------------------------- run
def preflight(a):
    cand, ctrl = CAND_DIR / "llama-server", CTRL_DIR / "llama-server"
    problems = []
    for b in (cand, ctrl):
        if not b.exists():
            problems.append(f"missing {b}")
    if problems:
        return dict(ok=False, problems=problems)
    impl_c, impl_k = CAND_DIR / "libllama-server-impl.so", CTRL_DIR / "libllama-server-impl.so"
    has_c = file_has(impl_c, MARK_DRAFT) if impl_c.exists() else file_has(cand, MARK_DRAFT)
    has_k = file_has(impl_k, MARK_DRAFT) if impl_k.exists() else file_has(ctrl, MARK_DRAFT)
    if not has_c:
        problems.append("candidate does not contain the #30574 draft-state save string: not the champion+#30574 build")
    if has_k:
        problems.append("control already contains the #30574 draft-state string: not an unported control")
    link = {}
    for nm, b in (("cand", cand), ("ctrl", ctrl)):
        r = subprocess.run([M.LINKAGE, str(b)], capture_output=True, text=True)
        link[nm] = r.returncode
        if r.returncode != 0:
            problems.append(f"verify_ggml_linkage failed for {nm}")
    prov = dict(candidate=dict(path=str(cand), sha256=sha256(cand), impl_sha256=(sha256(impl_c) if impl_c.exists() else None), commit="dd3d9721ca37321ad9159520bbd9616e3d0d6228",
                               label="champion e33229348 + #26004 (1c999e15b) + #30574 (deb387b8b), build 10345, fold CANDIDATE (not production, not yet the champion)"),
                control=dict(path=str(ctrl), sha256=sha256(ctrl), impl_sha256=(sha256(impl_k) if impl_k.exists() else None), label="CPU champion image 4348de400 (before the slot-file ports)"),
                linkage_rc=link)
    return dict(ok=not problems, problems=problems, provenance=prov)


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = preflight(a)
    (out / "preflight.json").write_text(json.dumps(pre, indent=1))
    if not pre["ok"]:
        print("PREFLIGHT FAILED:", pre["problems"], flush=True)
        return 6
    state = load_state(a)
    cal = A.Calibrator.from_rows([dict(model=MODEL_KEY, kind=v["kind"], unit="quarter", result="OK", est_raw_s=v["est_raw_s"], wall_s=v["wall_s"])
                                  for v in state.values() if v.get("rc") == 0])
    budget = A.Budget(a.stop_file, a.deadline_epoch)
    stages = build_stages(a)
    verdict = None
    for s in stages:
        if s.name.startswith(("bank_", "cells_")):
            if verdict is None:
                verdict = compute_verdict(a)
                (out / "VERDICT.json").write_text(json.dumps(verdict, indent=1))
                print("VERDICT", json.dumps({k: verdict[k] for k in ("candidate_passes", "control_discriminates", "proceed")}), flush=True)
            if not verdict["proceed"]:
                print(f"skip {s.name}: gates did not clear the candidate", flush=True)
                continue
        rc = run_stage(s, a, state, cal, budget)
        if rc == 9:
            break
        if rc != 0:
            print(f"{s.name} failed rc={rc}: stopping", flush=True)
            return 1
    if verdict is None and all(state.get(n, {}).get("rc") == 0 for n in ("gates_cand",) + (() if a.skip_control else ("gates_control",))):
        verdict = compute_verdict(a)
        (out / "VERDICT.json").write_text(json.dumps(verdict, indent=1))
    return finish(a, state, verdict, pre)


def compute_verdict(a):
    out = Path(a.out)
    cand = json.loads((out / "gates_cand" / "gates.json").read_text())
    vc = gate_verdict(cand)
    ns, nr = count_markers(out / "gates_cand")
    vc["draft_state_save_lines"], vc["draft_state_restore_lines"] = ns, nr
    ran = ns > 0 and nr > 0
    if not ran:
        vc["failed"].append("server log shows no draft-state save/restore line: the #30574 path did not run (vacuous gate)")
        vc["passed"] = False
    if a.skip_control:
        ctrl_path, ctrl = str(STUDY_V10_GATES), json.loads(STUDY_V10_GATES.read_text())
    else:
        ctrl_path, ctrl = str(out / "gates_control" / "gates.json"), json.loads((out / "gates_control" / "gates.json").read_text())
    vk = gate_verdict(ctrl)
    ck_acc_fail = any("acceptance ratio" in f for f in vk["failed"])         # decode may be missing on the old record; acceptance is the discriminating leg
    cs, cr = count_markers(out / "gates_control") if not a.skip_control else (0, 0)
    return dict(candidate=vc, candidate_passes=vc["passed"], control=vk, control_source=ctrl_path, control_discriminates=ck_acc_fail,
                control_draft_lines=(cs, cr), proceed=bool(vc["passed"] and ck_acc_fail),
                note="control_discriminates = the unported build shows the acceptance drop the rule is meant to catch; if it does not, the gate is non-discriminating and nothing is concluded")


def summarize_cells(a):
    rows = []
    for p in sorted(Path(a.out).glob("cells_*/summary.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("cell"):
                    rows.append({k: r.get(k) for k in ("launch", "cell", "result", "np", "unified", "mtp", "L", "source", "ttft_s_med", "decode_tps_stream_med",
                                                        "decode_tps_sum_med", "draft_n_by_wave", "draft_n_accepted_by_wave", "n_reprefill_by_wave", "binary_sha256")})
    return rows


def finish(a, state, verdict, pre):
    out = Path(a.out)
    cells = summarize_cells(a)
    done_all = all(v.get("rc") == 0 for v in state.values()) and bool(state)
    expected = [s.name for s in build_stages(a)]
    if verdict is not None and not verdict["proceed"]:
        expected = [n for n in expected if not n.startswith(("bank_", "cells_"))]
    complete = verdict is not None and all(state.get(n, {}).get("rc") == 0 for n in expected)
    rec = dict(schema=SCHEMA, item="kvu12-2-mtp-bank-cells", time=time.strftime("%FT%TZ", time.gmtime()), complete=complete, stages=state,
               verdict=None if verdict is None else {k: verdict[k] for k in ("candidate_passes", "control_discriminates", "proceed")},
               failed_reasons=None if verdict is None else verdict["candidate"]["failed"], provenance=pre.get("provenance"), n_cells=len(cells), cells=cells,
               labelling=("MTP-on bank decode tok/s are INDICATIVE and labelled with the candidate binary sha256 in provenance.candidate; "
                          "v10-binary MTP-on cells stay 'restored-draft'; a bank written by this build is NOT restorable on v10"),
               cpus=a.cpu_list, threads=a.threads, lens=a.lens, waves=a.waves)
    (out / "RUN_RECORD.json").write_text(json.dumps(rec, indent=1))
    print(f"RUN_RECORD complete={complete} verdict={rec['verdict']} cells={len(cells)}", flush=True)
    return 0 if (complete or not any(v.get("rc") not in (0, None) for v in state.values())) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("plan", "run"):
        p = sp.add_parser(nm)
        p.add_argument("--out", required=True)
        p.add_argument("--cpu-list", default="0-95")
        p.add_argument("--threads", type=int, default=96)
        p.add_argument("--port", type=int, default=18972)
        p.add_argument("--lens", default="8k", help="bank/cell context classes; 32k adds ~48 min of bank prefill per layout at 24 threads")
        p.add_argument("--waves", type=int, default=3)
        p.add_argument("--n-cells", type=int, default=4, help="how many of the 4 (layout, np) MTP-on bank cells to run, in the BANK_CELLS order")
        p.add_argument("--skip-control", action="store_true", help="use the 2026-10-20 v10 gates.json as the control record instead of re-running an unported build")
        p.add_argument("--stop-file", default="")
        p.add_argument("--deadline-epoch", type=int, default=0, help="no new stage that cannot finish before this (filler: grant end - 90 s)")
        p.add_argument("--hard-end-epoch", type=int, default=0, help="TERM the running driver at this time (filler: grant end)")
    a = ap.parse_args(argv)
    if a.port in range(8000, 8201):
        sys.exit("refusing production port range 8000-8200")
    return cmd_plan(a) if a.cmd == "plan" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
