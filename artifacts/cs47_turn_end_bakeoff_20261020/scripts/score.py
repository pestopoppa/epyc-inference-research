"""CS-47 accuracy-leg scorer. python -I score.py OUTDIR. Selection sweep -> op points -> confirmation touched ONCE."""
import json, glob, sys, random, math
from pathlib import Path
import numpy as np
T = Path("/mnt/raid0/llm/tmp/cs-47-20261020"); OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=True, parents=True)
D = Path("/mnt/raid0/llm/worktrees/cs47-research-20261020/data/turn-taking-cs45-v1/manifest")
clips = {c["clip_id"]: c for c in map(json.loads, (D / "clips.jsonl").read_text().splitlines())}
pairs = {p["pair_id"]: p for p in map(json.loads, (D / "pairs.jsonl").read_text().splitlines())}
DT = 512 / 16000

def silence_fires(P, thr, start_s, stop_s):
    f = []; cons = 0; ins = False; last = 0.0
    for i, p in enumerate(P):
        t = (i + 1) * DT
        if p >= thr:
            cons += 1
            if cons * DT >= start_s: ins = True
            if ins: last = t
        else:
            cons = 0
            if ins and t - last >= stop_s - 1e-9:
                f.append(round(t, 4)); ins = False
    return f

# ---- load raw
silero = {}; st = {}; pk = {}
for fn in glob.glob(str(T / "out/silero_s4_*.jsonl")) + glob.glob(str(T / "out/silero_*.jsonl")):
    for l in open(fn):
        r = json.loads(l); silero[r["clip"]] = r["probs"]
for fn in glob.glob(str(T / "out/st_*.jsonl")):
    for l in open(fn):
        r = json.loads(l); st[r["clip"]] = r["calls"]
for fn in glob.glob(str(T / "out/pk_*.jsonl")):
    for l in open(fn):
        r = json.loads(l); pk.setdefault(r["variant"], {})[r["clip"]] = r

# ---- configs: name -> (arm, tag, gate_s, fire_fn(clip)->list)
CFG = {}
for Tsec in (0.3, 0.5, 0.75, 1.0, 1.2):
    CFG[f"silero_silence_T{Tsec}"] = dict(arm="silero+silence", ref=False, gate=Tsec, fn=lambda c, Tsec=Tsec: silence_fires(silero[c], 0.5, 0.1, Tsec) if c in silero else None)
CFG["REF_nemo_voice_agent_conf0.6_start0.1_stop1.2"] = dict(arm="silero+silence", ref=True, gate=1.2, fn=lambda c: silence_fires(silero[c], 0.6, 0.1, 1.2) if c in silero else None)
for g in (0.2, 0.4, 0.6, 0.8, 1.0):
    for th in (0.0082, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        def fn(c, g=g, th=th):
            if c not in st: return None
            return sorted({round(r["t_fire_s"], 4) for r in st[c] if abs(r["gate_s"] - g) < 1e-9 and r["prob"] >= th})
        CFG[f"smartturn_v3.2_cpu_int8_gate{g}_theta{th}"] = dict(arm="smartturn", ref=(g == 1.0 and th == 0.0082), gate=g, fn=fn)
for var, rows in pk.items():
    for fb in (True, False):
        def fn(c, var=var, fb=fb):
            r = pk[var].get(c)
            if r is None or c not in silero: return None
            f = [e.get("released_t", e["t"]) for e in r["events"] if e["kind"] == "eou"]
            if fb: f += silence_fires(silero[c], 0.5, 0.1, 1.2)
            return sorted(f)
        CFG[f"parakeet_eou_{var.replace(',', '_')}{'_fb1.2' if fb else '_nofb'}"] = dict(arm="parakeet_eou", ref=False, gate=(1.2 if fb else 0.0), fn=fn, pk_complete=len(rows))

def item(cid, fires):
    c = clips[cid]; g = c["gold"]; lab = c["label"]
    s0 = g["user_speech_start_s"]; send = g["user_speech_end_s"]
    if fires is None: return dict(status="missing", fires=None)
    fs = [x for x in fires if x >= s0]
    if lab == "complete":
        eg = g["eot_gold_s"]; prem = [x for x in fs if x < eg]; post = [x for x in fs if x >= eg]
        hit = bool(post) and not prem
        return dict(status="hit" if hit else ("premature" if prem else "miss"), fires=fs, lat=(post[0] - eg) if hit else None)
    return dict(status="false_cut" if fs else "ok", fires=fs, lat=(fs[0] - send) if fs else None)

def metrics(name, ids, items, nmain):
    cp = [i for i in ids if clips[i]["label"] == "complete"]; ic = [i for i in ids if clips[i]["label"] == "incomplete"]; bc = [i for i in ids if clips[i]["label"] == "backchannel"]
    # denominator: every manifest item; missing (arm not run / crashed) counts as miss / and for negatives as NOT counted as ok? -> counted as miss for positives, no-fire for negatives
    rec = np.mean([items[i]["status"] == "hit" for i in cp]) if cp else float("nan")
    fpr = np.mean([items[i]["status"] == "false_cut" for i in ic]) if ic else float("nan")
    ifpr = np.mean([items[i]["status"] == "false_cut" for i in bc]) if bc else float("nan")
    lat = [items[i]["lat"] for i in cp if items[i]["status"] == "hit"]
    q = (lambda p: float(np.percentile(lat, p))) if lat else (lambda p: float("nan"))
    ok_item = [(items[i]["status"] == "hit") if clips[i]["label"] == "complete" else (items[i]["status"] == "ok") for i in ids]
    pasr_item = float(np.mean(ok_item)) if ids else float("nan")
    pp = [p for p in pairs.values() if p["clips"][0] in items and p["clips"][0] in set(ids)]
    pasr_pair = float(np.mean([items[p["clips"][0]]["status"] == "hit" and items[p["clips"][1]]["status"] == "ok" for p in pp])) if pp else float("nan")
    return dict(config=name, n_items=len(ids), n_complete=len(cp), n_incomplete=len(ic), n_backchannel=len(bc),
                n_missing=sum(items[i]["status"] == "missing" for i in ids),
                eot_recall=float(rec), eot_fpr=float(fpr), int_fpr=float(ifpr),
                lat_p10=q(10), lat_p50=q(50), lat_p90=q(90), n_hit=len(lat),
                pasr_item=pasr_item, pasr_pair=pasr_pair, premature=sum(items[i]["status"] == "premature" for i in cp))

split = {cid: clips[cid]["split"] for cid in clips}
allids = sorted(clips)
sel = [i for i in allids if split[i] == "selection"]; conf = [i for i in allids if split[i] == "confirmation"]
rows_all = {}; items_all = {}
for name, cf in CFG.items():
    items = {i: item(i, cf["fn"](i)) for i in allids}
    items_all[name] = items
    # an arm that did not run for a clip is 'missing' -> miss in the denominator
sel_m = {n: metrics(n, sel, items_all[n], len(sel)) for n in CFG}
for n in CFG: sel_m[n].update(arm=CFG[n]["arm"], gate_s=CFG[n]["gate"], ref=CFG[n]["ref"])
# operating-point choice on SELECTION only: highest EOT recall at EOT FPR <= 0.10 (tie: lower p50 latency)
fams = {}
for n, m in sel_m.items(): fams.setdefault(m["arm"], []).append(n)
chosen = {}
for arm, names in fams.items():
    if arm == "parakeet_eou":
        groups = {}
        for n in names: groups.setdefault(n.rsplit("_fb", 1)[0] if "_fb1.2" in n else n.rsplit("_nofb", 1)[0], []).append(n)
        sub = list(groups.values())
    else: sub = [names]
    for g in sub:
        feas = [n for n in g if sel_m[n]["eot_fpr"] <= 0.10]
        if feas:
            best = max(feas, key=lambda n: (sel_m[n]["eot_recall"], -(sel_m[n]["lat_p50"] if sel_m[n]["lat_p50"] == sel_m[n]["lat_p50"] else 9)))
            chosen[best] = "feasible_at_fpr<=0.10"
        else:
            best = min(g, key=lambda n: (sel_m[n]["eot_fpr"], -sel_m[n]["eot_recall"]))
            chosen[best] = "INFEASIBLE: no config meets FPR<=0.10 on selection; min-FPR config shown"
refs = [n for n in CFG if CFG[n]["ref"]]
# confirmation touched ONCE: chosen op points + named reference rows + constant policies
conf_names = sorted(set(chosen) | set(refs))
conf_m = {n: metrics(n, conf, items_all[n], len(conf)) for n in conf_names}
cp_all = lambda ids: [i for i in ids if clips[i]["label"] == "complete"]
const = {}
for nm, ids in (("selection", sel), ("confirmation", conf)):
    n = len(ids); c = len(cp_all(ids))
    const[nm] = {"always_end": dict(pasr_item=c / n, pasr_pair=0.0, eot_recall=1.0, eot_fpr=1.0, int_fpr=1.0, lat_p50=0.0),
                 "never_end": dict(pasr_item=(n - c) / n, pasr_pair=0.0, eot_recall=0.0, eot_fpr=0.0, int_fpr=0.0, lat_p50=None)}
# matched-latency comparison (synthetic ranking only): nearest-median-latency silence row from the sweep (selection)
def boot_diff(a, b, ids, B=2000, seed=7):
    # paired cluster bootstrap over dialogues of false-cut-rate difference (a-b) on incomplete clips
    ic = [i for i in ids if clips[i]["label"] == "incomplete"]
    cl = {}
    for i in ic: cl.setdefault(clips[i]["dialogue_id"], []).append(i)
    keys = list(cl); rnd = random.Random(seed); ds = []
    fa = {i: items_all[a][i]["status"] == "false_cut" for i in ic}; fb = {i: items_all[b][i]["status"] == "false_cut" for i in ic}
    for _ in range(B):
        ss = [i for k in (rnd.choice(keys) for _ in keys) for i in cl[k]]
        ds.append(np.mean([fa[i] for i in ss]) - np.mean([fb[i] for i in ss]))
    return float(np.mean([fa[i] for i in ic]) - np.mean([fb[i] for i in ic])), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
silence_names = [n for n in CFG if CFG[n]["arm"] == "silero+silence" and not CFG[n]["ref"]]
matched = {}
for n in chosen:
    if CFG[n]["arm"] == "silero+silence": continue
    L = conf_m[n]["lat_p50"]
    if L != L: matched[n] = dict(note="no hits: median latency undefined"); continue
    cand = [s for s in silence_names if sel_m[s]["lat_p50"] == sel_m[s]["lat_p50"]]
    s = min(cand, key=lambda s: abs(sel_m[s]["lat_p50"] - L))
    d, ci = boot_diff(n, s, conf)
    matched[n] = dict(silence_row=s, silence_row_p50_sel=sel_m[s]["lat_p50"], arm_p50_conf=L, latency_within_plus100ms=bool(L <= sel_m[s]["lat_p50"] + 0.1),
                      fpr_diff_arm_minus_silence_conf=d, boot95=ci, ci_excludes_zero_below=bool(ci[1] < 0))
res = dict(config_echo=dict(corpus="turn-taking-cs45-v1 v2 (research cb6b08fea)", thresholds=dict(matched_latency_ms=100, cpu_p95_ms=100, fpr_budget=0.10),
                           advance_eligible_strata=["human_recorded"], human_recorded_n=0,
                           n_clips=len(allids), n_selection=len(sel), n_confirmation=len(conf),
                           parakeet_variants=sorted(pk), smartturn_clips=len(st), silero_clips=len(silero),
                           sil_state_machine="p>=thr, start=0.1s sustained speech, stop=T s after last speech frame; frame 32 ms; decision at frame end"),
           selection=sel_m, chosen_op_points=chosen, confirmation=conf_m, constants=const, matched_latency=matched)
(OUT / "results.json").write_text(json.dumps(res, indent=1))
with open(OUT / "items.jsonl", "w") as fo:
    for n in sorted(CFG):
        for i in allids:
            it = items_all[n][i]; fo.write(json.dumps(dict(config=n, clip=i, split=split[i], label=clips[i]["label"], family=clips[i]["family"], **it)) + "\n")
def fmt(m): return f"{m['eot_recall']:.3f} | {m['eot_fpr']:.3f} | {m['int_fpr']:.3f} | {m['lat_p10']*1e3:.0f}/{m['lat_p50']*1e3:.0f}/{m['lat_p90']*1e3:.0f} | {m['pasr_item']:.3f} | {m['pasr_pair']:.3f} | {m['n_missing']}"
with open(OUT / "tables.md", "w") as f:
    f.write("SELECTION sweep (arm | config | recall | EOT-FPR | INT-FPR | lat p10/p50/p90 ms | PASR item | PASR pair | missing)\n")
    for n in sorted(CFG, key=lambda n: (CFG[n]["arm"], n)):
        f.write(f"{sel_m[n]['arm']} | {n}{' *' if n in chosen else ''} | {fmt(sel_m[n])}\n")
    f.write("\nCONFIRMATION (touched once)\n")
    for n in conf_names: f.write(f"{n} | {chosen.get(n, 'named reference')} | {fmt(conf_m[n])}\n")
print((OUT / "tables.md").read_text()); print(json.dumps(matched, indent=1))
