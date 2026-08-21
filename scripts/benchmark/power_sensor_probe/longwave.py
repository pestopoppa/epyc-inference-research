"""Run 3: long phases for the AVERAGED field's t_d/t_r/t_f (it has multi-second memory,
so 1 s phases never settle) + a 500 Hz arm at true Nyquist to bracket the aliasing knee."""
import ctypes, time, json, threading
import torch
rsmi = ctypes.CDLL("/opt/rocm/lib/librocm_smi64.so"); assert rsmi.rsmi_init(0)==0
_e=ctypes.c_uint64(); _r=ctypes.c_float(); _ts=ctypes.c_uint64(); _p=ctypes.c_uint64()
samples=[]; stop=threading.Event(); marks=[]
def sampler():
    while not stop.is_set():
        rsmi.rsmi_dev_energy_count_get(0, ctypes.byref(_e), ctypes.byref(_r), ctypes.byref(_ts))
        rsmi.rsmi_dev_power_ave_get(0, 0, ctypes.byref(_p))
        samples.append((time.perf_counter(), _e.value, _p.value))
def mark(l): marks.append((time.perf_counter(), l)); print(l, flush=True)
def busy_until(t_end, a, b):
    while time.perf_counter() < t_end:
        torch.mm(a, b); torch.cuda.synchronize()
def wave(period, dur, a, b):
    t=time.perf_counter(); end=t+dur
    while t < end:
        busy_until(min(t+period/2, end), a, b)
        t2=min(t+period, end)
        while time.perf_counter() < t2: pass
        t=t2
a=torch.randn(1024,1024,device='cuda',dtype=torch.float16); b=torch.randn_like(a)
torch.mm(a,b); torch.cuda.synchronize()
th=threading.Thread(target=sampler,daemon=True); th.start()
mark("idle0");    time.sleep(4)
mark("active");   busy_until(time.perf_counter()+12, a, b)
mark("recover");  time.sleep(15)
mark("wave_500hz"); wave(0.002, 8.0, a, b)
mark("idle_end"); time.sleep(3)
mark("end"); stop.set(); th.join(timeout=1)
json.dump({"marks":marks,"n":len(samples),"samples":samples}, open("trace_long.json","w"))
print(f"samples {len(samples)}")
