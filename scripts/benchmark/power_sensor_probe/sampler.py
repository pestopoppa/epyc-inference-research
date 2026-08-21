"""RVP-PWR-2/PWR-5 sampler — ctypes on librocm_smi64, bypassing the ~200 ms CLI spawn cost.
Samples BOTH the cumulative energy counter (the trusted instrument) and the averaged power
field (the filtered one under test) as fast as the API allows."""
import ctypes, time, json, sys

rsmi = ctypes.CDLL("/opt/rocm/lib/librocm_smi64.so")
assert rsmi.rsmi_init(0) == 0

def read_energy():
    e = ctypes.c_uint64(); res = ctypes.c_float(); ts = ctypes.c_uint64()
    st = rsmi.rsmi_dev_energy_count_get(0, ctypes.byref(e), ctypes.byref(res), ctypes.byref(ts))
    return st, e.value, res.value, ts.value

def read_power_avg():
    p = ctypes.c_uint64()
    st = rsmi.rsmi_dev_power_ave_get(0, 0, ctypes.byref(p))
    return st, p.value  # microwatts

def calibrate(n=2000):
    """Max sampling rate + does the energy counter tick at the documented 1 ms cadence?"""
    t0 = time.perf_counter()
    rows = []
    for _ in range(n):
        st, e, res, ts = read_energy()
        rows.append((time.perf_counter(), e, ts))
    t1 = time.perf_counter()
    per_call = (t1 - t0) / n * 1e6
    # distinct counter values / distinct device timestamps within the window
    evs = sorted({r[1] for r in rows}); tss = sorted({r[2] for r in rows})
    ts_deltas = [b - a for a, b in zip(tss, tss[1:])][:10]
    st, p = read_power_avg()
    return dict(per_call_us=round(per_call, 1), calls=n, wall_s=round(t1 - t0, 3),
                distinct_energy_values=len(evs), distinct_device_ts=len(tss),
                device_ts_deltas_first10=ts_deltas, lsb_uj=rows[0][2] if rows else None,
                counter_resolution=read_energy()[2], power_avg_now_w=p / 1e6)

if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=1))
