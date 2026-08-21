"""RVP-PWR-2/PWR-5: square-wave power-sensor characterisation on the MI210.
Phases: idle / 1s-1s wave (t_d,t_r,t_f) / 10 Hz wave (clean FFT case) / 250 Hz wave
(the paper's aliased case) / idle. Sampler thread records (t, energy_uJ, power_avg_uW)
via rsmi at max rate. Observations only (MEASUREMENT.md); no decision gates."""
import ctypes, time, json, threading, sys
import torch

rsmi = ctypes.CDLL("/opt/rocm/lib/librocm_smi64.so")
assert rsmi.rsmi_init(0) == 0
_e = ctypes.c_uint64(); _r = ctypes.c_float(); _ts = ctypes.c_uint64(); _p = ctypes.c_uint64()

samples = []
stop = threading.Event()
marks = []  # (t, label)

def sampler():
    while not stop.is_set():
        rsmi.rsmi_dev_energy_count_get(0, ctypes.byref(_e), ctypes.byref(_r), ctypes.byref(_ts))
        rsmi.rsmi_dev_power_ave_get(0, 0, ctypes.byref(_p))
        samples.append((time.perf_counter(), _e.value, _p.value))

def mark(label):
    marks.append((time.perf_counter(), label))
    print(f"{time.strftime('%H:%M:%S')} {label}", flush=True)

def busy_until(t_end, a, b):
    # sync EVERY op: the async queue must never backlog past the phase edge,
    # or the realized period stretches (measured: exactly halved frequency)
    while time.perf_counter() < t_end:
        torch.mm(a, b)
        torch.cuda.synchronize()

def wave(period_s, duration_s, a, b):
    half = period_s / 2
    t = time.perf_counter()
    end = t + duration_s
    while t < end:
        busy_until(min(t + half, end), a, b)
        # idle half: spin-wait cheap (sleep jitter breaks short periods)
        t2 = min(t + period_s, end)
        while time.perf_counter() < t2:
            if period_s > 0.05: time.sleep(0.001)
        t = t2

def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_properties(0)
    assert 'gfx90a' in getattr(dev, 'gcnArchName', ''), 'refuse: not gfx90a'
    a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    torch.mm(a, b); torch.cuda.synchronize()  # warm the kernel cache
    th = threading.Thread(target=sampler, daemon=True); th.start()
    mark("idle0");        time.sleep(3)
    mark("wave_1s_1s");   wave(2.0, 16.0, a, b)
    mark("idle1");        time.sleep(2)
    mark("wave_10hz");    wave(0.100, 8.0, a, b)
    mark("idle2");        time.sleep(2)
    mark("wave_250hz");   wave(0.004, 8.0, a, b)
    mark("idle3");        time.sleep(3)
    mark("end")
    stop.set(); th.join(timeout=1)
    with open("trace.json", "w") as f:
        json.dump({"marks": marks, "n_samples": len(samples),
                   "samples": samples}, f)
    print(f"samples: {len(samples)}; span {samples[-1][0]-samples[0][0]:.1f} s", flush=True)

if __name__ == "__main__":
    main()
