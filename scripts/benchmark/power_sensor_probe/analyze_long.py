import json, math
import numpy as np
RES=15.300000190734863
d=json.load(open("trace_long.json")); marks={l:t for t,l in d["marks"]}
s=np.array(d["samples"]); t,e,p=s[:,0],s[:,1],s[:,2]/1e6
BIN=0.002
edges=np.arange(t[0],t[-1],BIN); idx=np.unique(np.searchsorted(t,edges)[np.searchsorted(t,edges)<len(t)])
tb,eb=t[idx],e[idx]; dp=np.diff(eb)*RES/np.diff(tb)/1e6; tp=(tb[:-1]+tb[1:])/2
out={}
mi=(tp>=marks["idle0"])&(tp<marks["active"]); lo=float(np.median(dp[mi]))
ma=(tp>=marks["active"]+2)&(tp<marks["recover"]-0.5); hi=float(np.median(dp[ma]))
out["idle_W"]=round(lo,1); out["active_plateau_W"]=round(hi,1)
th10,th90=lo+.1*(hi-lo),lo+.9*(hi-lo)
on=marks["active"]; off=marks["recover"]
# averaged field rise
m=(t>=on)&(t<off); ta,pa=t[m],p[m]
out["avg_t_d_ms"]=round(float(ta[np.argmax(pa>=th10)]-on)*1e3,1) if (pa>=th10).any() else None
if (pa>=th10).any() and (pa>=th90).any():
    out["avg_t_r_10_90_ms"]=round(float(ta[np.argmax(pa>=th90)]-ta[np.argmax(pa>=th10)])*1e3,1)
else: out["avg_t_r_10_90_ms"]=f"th90 NEVER reached in 12 s (peak {float(pa.max()):.0f} W of {hi:.0f} W)"
# averaged field fall
m=(t>=off)&(t<marks["wave_500hz"]); ta,pa=t[m],p[m]
if (pa<=th90).any() and (pa<=th10).any():
    out["avg_t_f_90_10_ms"]=round(float(ta[np.argmax(pa<=th10)]-ta[np.argmax(pa<=th90)])*1e3,1)
    out["avg_settle_to_th10_after_off_ms"]=round(float(ta[np.argmax(pa<=th10)]-off)*1e3,1)
# derived rise/fall
m=(tp>=on)&(tp<off); td_,dpd=tp[m],dp[m]
out["derived_t_d_ms"]=round(float(td_[np.argmax(dpd>=th10)]-on)*1e3,1)
out["derived_t_r_10_90_ms"]=round(float(td_[np.argmax(dpd>=th90)]-td_[np.argmax(dpd>=th10)])*1e3,1)
# 500 Hz FFT
m=(tp>=marks["wave_500hz"])&(tp<marks["idle_end"]); x=dp[m]-dp[m].mean(); n=len(x)
fr=np.fft.rfftfreq(n,d=BIN); mag=np.abs(np.fft.rfft(x*np.hanning(n))); band=fr>1
fpk=float(fr[band][np.argmax(mag[band])])
mask=band.copy()
for h in range(1,4): mask &= ~(np.abs(fr-500*h)<75)
floor=float(np.median(mag[mask])); pk=float(mag[band].max())
out["fft_500hz"]=dict(commanded=500.0, peak_hz=round(fpk,1),
                      shift_pct=round(100*(fpk-500)/500,1),
                      peak_over_floor_db=round(20*math.log10(pk/max(floor,1e-9)),1),
                      note="BIN=2ms puts Nyquist at 250 Hz for the DERIVED trace - a 500 Hz command MUST alias here; the question is the signature")
dt=np.diff(t); out["sampler_median_us"]=round(float(np.median(dt))*1e6,1)
print(json.dumps(out,indent=1)); json.dump(out,open("analysis_long.json","w"),indent=1)
