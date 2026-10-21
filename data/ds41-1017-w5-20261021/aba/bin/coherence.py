#!/usr/bin/env python3
"""usage: coherence.py ARM CPULIST NTHREADS OUTDIR PORT  -- launches llama-server (own PID), runs greedy prompts, kills it."""
import json, os, subprocess, sys, time, urllib.request, hashlib
arm, cpus, nt, out, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
B = "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin"
MD = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf"
M = {"1017": f"{MD}/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf",
     "sept-q4": f"{MD}/DeepSeek-V4.1-Flash-Q4.gguf"}[arm]
env = dict(os.environ, OMP_PROC_BIND="spread", OMP_PLACES="cores", OMP_WAIT_POLICY="active",
           OMP_DYNAMIC="false", GGML_IQK="1", LD_LIBRARY_PATH=B)
argv = ["taskset", "-c", cpus, "numactl", "--interleave=all", f"{B}/llama-server", "-m", M, "-np", "1",
        "-c", "8192", "-t", nt, "-tb", nt, "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16",
        "--device", "none", "-ngl", "0", "-fa", "on", "--host", "127.0.0.1", "--port", str(port),
        "--no-webui", "--metrics"]
os.makedirs(out, exist_ok=True)
json.dump({"argv": argv, "env": {k: env[k] for k in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC", "GGML_IQK", "LD_LIBRARY_PATH")}},
          open(f"{out}/coh-{arm}.recipe.json", "w"), indent=1)
log = open(f"{out}/coh-{arm}.server.log", "w")
p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
def call(path, body=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
try:
    t0 = time.time()
    while True:
        if p.poll() is not None: raise SystemExit(f"server exited rc={p.returncode}")
        try:
            if call("/health", timeout=5).get("status") == "ok": break
        except Exception: pass
        if time.time() - t0 > 3600: raise SystemExit("load timeout")
        time.sleep(5)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    prompts = [json.loads(l) for l in open(os.environ.get("COH_PROMPTS","/mnt/raid0/llm/tmp/ds41-w5-20261021/bin/prompts.jsonl"))]
    rows = []
    for pr in prompts:
        r = call("/completion", {"prompt": pr["prompt"], "n_predict": pr.get("n_predict", 200), "temperature": 0,
                                 "top_k": 1, "seed": 42, "return_tokens": True, "cache_prompt": False})
        text = r["content"]
        rows.append({"id": pr["id"], "text": text, "token_ids": r.get("tokens"), "finish": r.get("stop_type"),
                     "http_ok": True, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "prompt_tokens": r.get("tokens_evaluated"), "timings": r.get("timings")})
        print(pr["id"], r.get("tokens_evaluated"), repr(text[:160]), flush=True)
    with open(f"{out}/coh-{arm}{os.environ.get("COH_TAG","")}.rows.jsonl", "w") as fh:
        for row in rows: fh.write(json.dumps(row) + "\n")
finally:
    p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait(60)
    print("server pid", p.pid, "rc", p.returncode)
