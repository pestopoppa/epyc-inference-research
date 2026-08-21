"""Phase-lock analysis for a decode trace:
1. steady decode window = [launch + settle, proc_exit - guard], settle > measured t_r (5 s)
2. ground truth: (E_end - E_start) * 15.3 / dt  -> true mean W (immune to filtering)
3. averaged field over same window: mean, std, drift
4. FFT of dE/dt at 1 ms bins: peak near the token cadence from llama's own timing
"""
import json, math, sys
import numpy as np
RES = 15.300000190734863
tag = sys.argv[1]
d = json.load(open(f"trace_{tag}.json"))
marks = {l: t for t, l in d["marks"]}
s = np.array(d["samples"]); t, e, p = s[:,0], s[:,1], s[:,2]/1e6
timing = d["decode_timing"]

out = {"tag": tag, "tok_s": timing.get("tok_s"), "n_tok": timing.get("n")}
# steady window: decode runs from ~launch+load to proc_exit; settle 6 s after launch
# find decode start better: first time derived power rises above idle+30 W sustained
BIN = 0.001
edges = np.arange(t[0], t[-1], BIN)
idx = np.unique(np.searchsorted(t, edges)[np.searchsorted(t, edges) < len(t)])
tb, eb = t[idx], e[idx]
dp = np.diff(eb) * RES / np.diff(tb) / 1e6
tp = (tb[:-1] + tb[1:]) / 2
idle_w = float(np.median(dp[(tp >= marks["idle_pre"]) & (tp < marks["launch"])]))
out["idle_W"] = round(idle_w, 1)
hot = tp[(dp > idle_w + 30) & (tp > marks["launch"]) & (tp < marks["proc_exit"])]
if len(hot) < 100: print(json.dumps({"error": "no sustained load found", **out})); sys.exit(1)
w0, w1 = hot[0] + 6.0, marks["proc_exit"] - 1.0   # settle past t_r(avg)~4.2s
out["window_s"] = round(w1 - w0, 1)
if w1 - w0 < 5: print(json.dumps({"error": "window too short", **out})); sys.exit(1)

# ground truth from energy endpoints
i0, i1 = np.searchsorted(t, w0), np.searchsorted(t, w1)
true_W = (e[i1] - e[i0]) * RES / (t[i1] - t[i0]) / 1e6
out["true_mean_W"] = round(float(true_W), 2)
# averaged field over the same window
m = (t >= w0) & (t <= w1)
out["avg_field_mean_W"] = round(float(p[m].mean()), 2)
out["avg_field_std_W"] = round(float(p[m].std()), 2)
out["discrepancy_W"] = round(float(p[m].mean() - true_W), 2)
out["discrepancy_pct"] = round(100 * float(p[m].mean() - true_W) / float(true_W), 2)
# stability: is the discrepancy STABLE (phase-lock signature) or noisy?
seg = np.array_split(np.where(m)[0], 8)
per = []
for si in seg:
    tw = (e[si[-1]] - e[si[0]]) * RES / (t[si[-1]] - t[si[0]]) / 1e6
    per.append(float(p[si].mean() - tw))
out["discrepancy_per_octant_W"] = [round(x, 2) for x in per]
out["discrepancy_octant_std_W"] = round(float(np.std(per)), 2)

# FFT of derived power in the window: token-cadence peak?
mm = (tp >= w0) & (tp <= w1)
x = dp[mm] - dp[mm].mean(); n = len(x)
fr = np.fft.rfftfreq(n, d=BIN); mag = np.abs(np.fft.rfft(x * np.hanning(n)))
band = fr > 2.0
fpk = float(fr[band][np.argmax(mag[band])])
out["fft_peak_hz"] = round(fpk, 1)
if timing.get("tok_s"):
    out["expected_cadence_hz"] = round(timing["tok_s"], 1)
    out["peak_vs_cadence_pct"] = round(100 * (fpk - timing["tok_s"]) / timing["tok_s"], 1)
    # spectral magnitude AT the cadence vs floor
    at = float(mag[np.argmin(np.abs(fr - timing["tok_s"]))])
    mask = band.copy()
    for h in range(1, 5): mask &= ~(np.abs(fr - timing["tok_s"] * h) < 0.1 * timing["tok_s"])
    floor = float(np.median(mag[mask]))
    out["cadence_peak_over_floor_db"] = round(20 * math.log10(at / max(floor, 1e-9)), 1)
print(json.dumps(out, indent=1))
json.dump(out, open(f"phaselock_{tag}.json", "w"), indent=1)
