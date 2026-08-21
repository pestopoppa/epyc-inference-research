import json, math
import numpy as np

d = json.load(open("trace.json"))
marks = {l: t for t, l in d["marks"]}
s = np.array(d["samples"])  # t, energy_uJ, power_uW
t, e, p = s[:,0], s[:,1], s[:,2] / 1e6  # power in W

# derived power: bin energy at 2 ms, dE/dt
BIN = 0.002
t0, t1 = t[0], t[-1]
edges = np.arange(t0, t1, BIN)
idx = np.searchsorted(t, edges)
idx = np.unique(idx[idx < len(t)])
tb, eb = t[idx], e[idx]
RES_UJ = 15.300000190734863  # counter LSB from rsmi counter_resolution: energy_uJ = counter * RES
dp = np.diff(eb) * RES_UJ / np.diff(tb) / 1e6      # W
tp = (tb[:-1] + tb[1:]) / 2

def window(name_a, name_b):
    m = (tp >= marks[name_a]) & (tp < marks[name_b])
    return tp[m], dp[m]

def avg_window(name_a, name_b):
    m = (t >= marks[name_a]) & (t < marks[name_b])
    return t[m], p[m]

out = {}
# --- baseline levels
_, dpi = window("idle0", "wave_1s_1s"); out["idle_W_derived"] = round(float(np.median(dpi)), 1)
tw, dpw = window("wave_1s_1s", "idle1")
out["active_W_derived_p90"] = round(float(np.percentile(dpw, 90)), 1)

# --- t_d / t_r / t_f of the AVERAGED field, per 1s/1s cycle
lo, hi = out["idle_W_derived"], out["active_W_derived_p90"]
th10, th90 = lo + 0.1*(hi-lo), lo + 0.9*(hi-lo)
rises, falls, delays = [], [], []
start = marks["wave_1s_1s"]
ta, pa = avg_window("wave_1s_1s", "idle1")
for k in range(8):
    on = start + k*2.0          # commanded rise
    off = on + 1.0              # commanded fall
    m = (ta >= on) & (ta < on + 1.0)
    if not m.any(): continue
    seg_t, seg_p = ta[m], pa[m]
    i10 = np.argmax(seg_p >= th10) if (seg_p >= th10).any() else None
    i90 = np.argmax(seg_p >= th90) if (seg_p >= th90).any() else None
    if i10 is not None and (seg_p >= th10).any():
        delays.append(seg_t[i10] - on)
    if i10 is not None and i90 is not None and (seg_p>=th90).any() and (seg_p>=th10).any():
        rises.append(seg_t[i90] - seg_t[i10])
    mf = (ta >= off) & (ta < off + 1.0)
    if mf.any():
        seg_t, seg_p = ta[mf], pa[mf]
        j90 = np.argmax(seg_p <= th90) if (seg_p <= th90).any() else None
        j10 = np.argmax(seg_p <= th10) if (seg_p <= th10).any() else None
        if j90 is not None and j10 is not None and (seg_p<=th10).any() and (seg_p<=th90).any():
            falls.append(seg_t[j10] - seg_t[j90])
out["avg_field_t_d_ms"] = dict(median=round(float(np.median(delays))*1e3,1), n=len(delays)) if delays else "NOT REACHED"
out["avg_field_t_r_10_90_ms"] = dict(median=round(float(np.median(rises))*1e3,1), n=len(rises)) if rises else "th90 NEVER REACHED within 1 s active phase"
out["avg_field_t_f_90_10_ms"] = dict(median=round(float(np.median(falls))*1e3,1), n=len(falls)) if falls else "n/a"
out["avg_field_peak_W_in_wave"] = round(float(pa.max()),1)
out["avg_field_reaches_th90"] = bool((pa >= th90).any())

# same transitions on DERIVED power
dr, dd = [], []
for k in range(8):
    on = start + k*2.0
    m = (tp >= on) & (tp < on + 1.0)
    if not m.any(): continue
    seg_t, seg_p = tp[m], dpw[(tw >= on) & (tw < on+1.0)] if False else (tp[m], dp[(tp >= on) & (tp < on+1.0)])
    seg_t, seg_p = seg_t, dp[(tp >= on) & (tp < on+1.0)]
    if (seg_p >= th10).any(): dd.append(seg_t[np.argmax(seg_p >= th10)] - on)
    if (seg_p >= th10).any() and (seg_p >= th90).any():
        dr.append(seg_t[np.argmax(seg_p >= th90)] - seg_t[np.argmax(seg_p >= th10)])
out["derived_t_d_ms"] = dict(median=round(float(np.median(dd))*1e3,1), n=len(dd)) if dd else "n/a"
out["derived_t_r_10_90_ms"] = dict(median=round(float(np.median(dr))*1e3,1), n=len(dr)) if dr else "n/a"

# --- FFT of derived power for the two wave windows
def fft_peak(name_a, name_b, commanded_hz):
    tw_, dw_ = window(name_a, name_b)
    x = dw_ - dw_.mean()
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=BIN)
    mag = np.abs(np.fft.rfft(x * np.hanning(n)))
    band = (freqs > 1.0)
    fpk = float(freqs[band][np.argmax(mag[band])])
    # noise floor: median magnitude excluding +/-15% around commanded + harmonics
    mask = np.ones_like(freqs, bool); mask[~band] = False
    for h in range(1, 6):
        mask &= ~(np.abs(freqs - commanded_hz*h) < 0.15*commanded_hz)
    floor = float(np.median(mag[mask])); peak = float(mag[band].max())
    return dict(commanded_hz=commanded_hz, peak_hz=round(fpk,2),
                peak_shift_pct=round(100*(fpk-commanded_hz)/commanded_hz,1),
                peak_over_floor_db=round(20*math.log10(peak/max(floor,1e-9)),1))
out["fft_10hz"] = fft_peak("wave_10hz", "idle2", 10.0)
out["fft_250hz"] = fft_peak("wave_250hz", "idle3", 250.0)

# sampling health
dt = np.diff(t); out["sampler"] = dict(median_us=round(float(np.median(dt))*1e6,1),
                                       p99_us=round(float(np.percentile(dt,99))*1e6,1),
                                       max_ms=round(float(dt.max())*1e3,2))
print(json.dumps(out, indent=1))
json.dump(out, open("analysis.json","w"), indent=1)
