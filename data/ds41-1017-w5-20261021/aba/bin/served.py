#!/usr/bin/env python3
"""usage: served.py ARM OUTDIR PORT -- N=1 no-drafter served decode, AK serving argv shape minus the drafter."""
import json, os, subprocess, sys, time, urllib.request
arm, out, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
SEG = os.environ.get("SEG", "x"); NREQ = int(os.environ.get("NREQ", "4")); GPUJOB = os.environ.get("GPUJOB", "none")
B = "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin"
MD = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf"
M = {"1017": f"{MD}/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf",
     "sept-q4": f"{MD}/DeepSeek-V4.1-Flash-Q4.gguf"}[arm]
env = dict(os.environ, OMP_PROC_BIND="spread", OMP_PLACES="cores", OMP_WAIT_POLICY="active",
           OMP_DYNAMIC="false", GGML_IQK="1", LD_LIBRARY_PATH=B)
argv = ["taskset", "-c", "0-95", "numactl", "--interleave=all", f"{B}/llama-server", "-m", M, "-np", "1",
        "-c", "8192", "-t", "48", "-tb", "48", "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16",
        "--device", "none", "-ngl", "0", "-fa", "on", "--host", "127.0.0.1", "--port", str(port),
        "--metrics", "--slots", "--no-kv-unified", "--no-mmap", "--no-webui", "--load-threads", "48"]
os.makedirs(out, exist_ok=True)
json.dump({"argv": argv, "env": {k: env[k] for k in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC", "GGML_IQK", "LD_LIBRARY_PATH")}},
          open(f"{out}/served-{SEG}-{arm}.recipe.json", "w"), indent=1)
log = open(f"{out}/served-{SEG}-{arm}.server.log", "w")
p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
def call(path, body=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
PROMPT = ("Write a detailed, well-organized technical explanation of how a modern CPU executes instructions out of order, "
          "covering register renaming, the reorder buffer, reservation stations, branch prediction and memory disambiguation.\n\n")
try:
    t0 = time.time()
    while True:
        if p.poll() is not None: raise SystemExit(f"server exited rc={p.returncode}")
        try:
            if call("/health", timeout=5).get("status") == "ok": break
        except Exception: pass
        if time.time() - t0 > 3600: raise SystemExit("load timeout")
        time.sleep(5)
    smi = subprocess.Popen(["bash", "-c", "while true; do date -u +%FT%TZ; rocm-smi --showuse --showmemuse 2>&1 | grep -E 'GPU\\[|use'; sleep 10; done"],
                           stdout=open(f"{out}/served-{SEG}-{arm}.rocmsmi.txt", "w"), stderr=subprocess.STDOUT)
    rows = []
    for i in range(NREQ):  # request 0 = warmup, excluded
        r = call("/completion", {"prompt": PROMPT, "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 42,
                                 "cache_prompt": False, "ignore_eos": True})
        t = r["timings"]
        rows.append({"i": i, "warmup": i == 0, "predicted_n": t["predicted_n"], "predicted_per_second": t["predicted_per_second"],
                     "prompt_n": t["prompt_n"], "prompt_per_second": t["prompt_per_second"], "content_head": r["content"][:200]})
        print(rows[-1], flush=True)
    smi.terminate(); smi.wait(30)
    json.dump({"arm": arm, "segment": SEG, "gpu_job": GPUJOB, "load_s": round(time.time()-t0), "rows": rows}, open(f"{out}/served-{SEG}-{arm}.json", "w"), indent=1)
finally:
    p.terminate()
    try: p.wait(120)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait(60)
    print("server pid", p.pid, "rc", p.returncode)
