"""RVP-PWR-2 follow-on: token-cadence phase-lock measurement on a REAL llama.cpp decode.
Runs llama-cli (production binary, its own libs) while sampling energy counter + averaged
power at ~107 us. Per run: (1) FFT of dE/dt -> is the token cadence visible as a spectral
peak; (2) ground-truth mean power (energy endpoints) vs the averaged field over the same
settled window -> discrepancy + stability; (3) cadence from llama's own timing line.
Observations only (MEASUREMENT.md). PID captured; llama-cli self-terminates at -n."""
import ctypes, time, json, threading, subprocess, sys, os, re

MODEL = sys.argv[1]; NTOK = int(sys.argv[2]); TAG = sys.argv[3]
BIN = "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-bench"
ENV = dict(os.environ, LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp/build-hip/bin")

rsmi = ctypes.CDLL("/opt/rocm/lib/librocm_smi64.so"); assert rsmi.rsmi_init(0)==0
_e=ctypes.c_uint64(); _r=ctypes.c_float(); _ts=ctypes.c_uint64(); _p=ctypes.c_uint64()
samples=[]; stop=threading.Event(); marks=[]
def sampler():
    while not stop.is_set():
        rsmi.rsmi_dev_energy_count_get(0, ctypes.byref(_e), ctypes.byref(_r), ctypes.byref(_ts))
        rsmi.rsmi_dev_power_ave_get(0, 0, ctypes.byref(_p))
        samples.append((time.perf_counter(), _e.value, _p.value))
def mark(l): marks.append((time.perf_counter(), l)); print(f"[{TAG}] {l}", flush=True)

th=threading.Thread(target=sampler,daemon=True); th.start()
mark("idle_pre"); time.sleep(3)
mark("launch")
cmd=[BIN, "-m", MODEL, "-dev", "ROCm0", "-ngl", "999", "-fa", "on",
     "-p", "0", "-n", str(NTOK), "-r", "2", "-o", "jsonl"]
t0=time.perf_counter()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV, text=True)
print(f"[{TAG}] pid={proc.pid}", flush=True)
out, _ = proc.communicate(timeout=600)
mark("proc_exit")
time.sleep(3); mark("idle_post")
stop.set(); th.join(timeout=1)

# llama-bench jsonl: avg_ts for the tg test
decode = {}
for l in out.splitlines():
    l=l.strip()
    if l.startswith('{'):
        try:
            j=json.loads(l)
            if j.get('n_gen'): decode=dict(tok_s=j.get('avg_ts'), n=j.get('n_gen'), stddev_ts=j.get('stddev_ts'))
        except Exception: pass
decode['raw']=[l for l in out.splitlines() if 'ROCm' in l or l.strip().startswith('{')][:6]
json.dump({"tag":TAG,"model":MODEL,"marks":marks,"n_samples":len(samples),
           "decode_timing":decode,"samples":samples},
          open(f"trace_{TAG}.json","w"))
print(f"[{TAG}] samples={len(samples)} timing={ {k:v for k,v in decode.items() if k!='raw'} }", flush=True)
rc=proc.poll(); print(f"[{TAG}] exit={rc}")
