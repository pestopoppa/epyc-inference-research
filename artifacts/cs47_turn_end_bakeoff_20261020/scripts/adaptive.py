"""CS-47 addendum: semantic-gated adaptive timeout, OFFLINE re-score of already-computed streams (no inference). python -I."""
import json, glob, sys
from pathlib import Path
import numpy as np
T = Path("/mnt/raid0/llm/tmp/cs-47-20261020")
D = Path("/mnt/raid0/llm/worktrees/cs47-research-20261020/data/turn-taking-cs45-v1/manifest")
clips = {c["clip_id"]: c for c in map(json.loads, (D / "clips.jsonl").read_text().splitlines())}
silero, st, pk = {}, {}, {}
for fn in glob.glob(str(T / "out/silero_*.jsonl")):
    for l in open(fn): r = json.loads(l); silero[r["clip"]] = r["probs"]
for fn in glob.glob(str(T / "out/st_*.jsonl")):
    for l in open(fn): r = json.loads(l); st[r["clip"]] = r["calls"]
for fn in glob.glob(str(T / "out/pk_*.jsonl")):
    for l in open(fn): r = json.loads(l); pk.setdefault(r["variant"], {})[r["clip"]] = r
DT = 512 / 16000
def clip_len(c): return clips[c]["gold"]["clip_end_s"]
def next_speech(P, ls):   # first speech frame end-time after last-speech time ls (thr .5)
    for i, p in enumerate(P):
        t = (i + 1) * DT
        if t > ls + 1e-9 and p >= 0.5: return t
    return 1e9
def st_adaptive(c, th, g, ts, tl):
    """returns (fires, censored_flag): censored = a low-p episode whose T_long wait runs past the clip end"""
    fires, cens = [], False
    P = silero[c]; L = clip_len(c)
    for r in st[c]:
        if abs(r["gate_s"] - g) > 1e-9: continue
        ls = r["sil_start_s"]; ns = next_speech(P, ls)
        if r["prob"] >= th: tf = ls + ts
        else: tf = ls + tl
        if tf >= ns: continue            # speech resumed first
        if tf > L: cens = True; continue
        fires.append(round(tf, 4))
    return sorted(fires), cens
def item(cid, fires, cens):
    c = clips[cid]; g = c["gold"]; s0 = g["user_speech_start_s"]; send = g["user_speech_end_s"]
    fs = [x for x in fires if x >= s0]
    if c["label"] == "complete":
        eg = g["eot_gold_s"]; prem = [x for x in fs if x < eg]; post = [x for x in fs if x >= eg]
        hit = bool(post) and not prem
        return dict(st="hit" if hit else ("premature" if prem else ("censored" if cens else "miss")), lat=(post[0] - eg) if hit else None)
    return dict(st="false_cut" if fs else ("never_fired_censored" if cens else "ok_held"), lat=(fs[0] - send) if fs else None)
def metrics(ids, it, ts):
    cp = [i for i in ids if clips[i]["label"] == "complete"]; ic = [i for i in ids if clips[i]["label"] == "incomplete"]; bc = [i for i in ids if clips[i]["label"] == "backchannel"]
    lat = [it[i]["lat"] for i in cp if it[i]["st"] == "hit"]
    q = lambda p: float(np.percentile(lat, p) * 1e3) if lat else None
    return dict(recall=float(np.mean([it[i]["st"] == "hit" for i in cp])), eot_fpr=float(np.mean([it[i]["st"] == "false_cut" for i in ic])),
                int_fpr=float(np.mean([it[i]["st"] == "false_cut" for i in bc])),
                incomplete_never_fired_censored=float(np.mean([it[i]["st"] == "never_fired_censored" for i in ic])),
                incomplete_held_not_censored=float(np.mean([it[i]["st"] == "ok_held" for i in ic])),
                backchannel_never_fired_censored=float(np.mean([it[i]["st"] == "never_fired_censored" for i in bc])),
                complete_censored_or_missed=float(np.mean([it[i]["st"] in ("censored", "miss", "premature") for i in cp])),
                complete_answered_within_Tshort=(float(np.mean([it[i]["st"] == "hit" and it[i]["lat"] <= ts + 0.05 for i in cp])) if ts else None),
                lat_p10_p50_p90_ms=[q(10), q(50), q(90)] if lat else None, n=len(ids), n_complete=len(cp), n_incomplete=len(ic), n_backchannel=len(bc))
sel = [i for i in sorted(clips) if clips[i]["split"] == "selection"]; conf = [i for i in sorted(clips) if clips[i]["split"] == "confirmation"]
rows = []
THS = (0.0082, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99)
for th in THS:
    for g in (0.2, 0.4):
        for ts in (0.3, 0.5):
            if g > ts: continue
            for tl in (2.5, 3.0, 5.0):
                its = {}
                for c in clips:
                    f, ce = st_adaptive(c, th, g, ts, tl); its[c] = item(c, f, ce)
                rows.append(dict(arm="smartturn_silero_adaptive", theta=th, gate_s=g, T_short=ts, T_long=tl, items=its))
# Parakeet: EOU token = complete signal, no fixed fallback below T_long (T_long >= 2.5 s is beyond the 2.0 s tail -> inert)
for var in sorted(pk):
    for tl in (2.5, 3.0, 5.0):
        its = {}
        for c in clips:
            r = pk[var].get(c); L = clip_len(c)
            f = sorted(e.get("released_t", e["t"]) for e in r["events"] if e["kind"] == "eou") if r else []
            # silence-fallback at T_long can never fire inside the clip (tail silence is 2.0 s): clip is censored if it never fired
            its[c] = item(c, f, cens=True)
        rows.append(dict(arm="parakeet_eou_adaptive", variant=var, T_long=tl, T_short=None, items=its))
for r in rows: r["sel"] = metrics(sel, r["items"], r["T_short"])
chosen = []
groups = {}
for r in rows:
    key = r["arm"] if r["arm"].startswith("smart") else (r["arm"], r["variant"])
    groups.setdefault(key, []).append(r)
for key, g in groups.items():
    feas = [r for r in g if r["sel"]["eot_fpr"] <= 0.10]
    if feas: b = max(feas, key=lambda r: (r["sel"]["recall"], -r["T_long"])); tag = "feasible"
    else: b = min(g, key=lambda r: (r["sel"]["eot_fpr"], -r["sel"]["recall"], r["T_long"])); tag = "INFEASIBLE(min-FPR shown)"
    b["conf"] = metrics(conf, b["items"], b["T_short"]); b["tag"] = tag; chosen.append(b)
out = []
for b in chosen:
    d = {k: v for k, v in b.items() if k != "items"}; out.append(d)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
with open(sys.argv[1] + ".items.jsonl", "w") as fo:
    for b in chosen:
        for c, it in b["items"].items(): fo.write(json.dumps(dict(arm=b["arm"], variant=b.get("variant"), theta=b.get("theta"), gate=b.get("gate_s"), T_short=b["T_short"], T_long=b["T_long"], clip=c, split=clips[c]["split"], **it)) + "\n")
for d in out: print(json.dumps(d))
# selection-only: best recall per theta at Smart Turn (to show the tradeoff)
print("SELECTION-ONLY feasibility scan (smartturn): theta g Ts Tl recall fpr")
for r in rows:
    if r["arm"].startswith("smart") and r["T_long"] == 2.5 and r["sel"]["eot_fpr"] <= 0.25: print(r["theta"], r["gate_s"], r["T_short"], round(r["sel"]["recall"], 3), round(r["sel"]["eot_fpr"], 3))
