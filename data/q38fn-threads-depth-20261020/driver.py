#!/usr/bin/env python3
"""Q38FN served thread-count x context-depth sweep (2026-10-20, operator-requested).

Question: does the CPU Flash-Next server want -t 48 or 96, and does -tb (which in this
binary governs EVERY ubatch with n_tokens > 1: prefill AND MTP verify) want 48 or 96,
at realistic agentic depths?

Method (prefill-once, as in the AK long-context audit C1):
  1. arm B0 server (-t48 -tb96) tokenizes wikitext2_test, then GROWS one slot through
     depths 512 -> 8k -> 32k -> 64k -> 128k, saving the slot at each depth.
  2. every arm restores each depth's slot and measures:
       A: prefix + next 4096 doc tokens, n_predict 1  -> prefill-at-depth (prompt t/s)
       B: prefix + fixed probe, 256 greedy tokens, MTP -> decode-at-depth (pred t/s), x2
     restoring before every request (the recurrent GDN layers cannot roll back).
  3. arms run B0, A, C, B1 (ABA drift control on B).
Every record is appended to results.jsonl as it lands.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request

R = "/mnt/raid0/llm/tmp/q38fn-threads-depth-20261020"
BIN = "/mnt/raid0/llm/kernels/builds/cpu-20261004-b0ba1d427/bin"
MODEL = ("/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
         "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
MTP = "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
CORPUS = "/mnt/raid0/llm/data/wikitext2_test.txt"
SLOTDIR = f"{R}/slots"
PORT = 18074
URL = f"http://127.0.0.1:{PORT}"
OUT = f"{R}/results.jsonl"

DEPTHS = [512, 8192, 32768, 65536, 131072]
TAIL = 4096
N_PRED = 256
B_REPS = 2
CTX = 131072 + TAIL + 1024

ENV_ADD = {
    "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false", "GGML_IQK": "1", "GGML_FA_SPLIT_KV": "0",
    "GGML_NOHUGEPAGE_PROCESS": "1", "GGML_FUSED_DECODE_OFF": "1",
    "LD_LIBRARY_PATH": f"{BIN}:" + os.environ.get("LD_LIBRARY_PATH", ""),
}
ARMS = [("B0", 48, 96), ("A", 48, 48), ("C", 96, 96), ("B1", 48, 96)]
if len(sys.argv) > 1:
    ARMS = [a for a in ARMS if a[0] in sys.argv[1].split(",")]
if os.environ.get("SMOKE"):
    DEPTHS, B_REPS, N_PRED = [512, 8192], 1, 64
    SLOTDIR, OUT = f"{R}/slots-smoke", f"{R}/smoke.jsonl"


def log(rec):
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "content"}), flush=True)


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def start(arm, t, tb):
    argv = ["taskset", "-c", "0-95", "numactl", "--interleave=all", f"{BIN}/llama-server",
            "--host", "127.0.0.1", "--port", str(PORT), "-m", MODEL, "-md", MTP,
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.5",
            "-np", "1", "-c", str(CTX), "-t", str(t), "-tb", str(tb), "--no-mmap",
            "--cache-ram", "0", "-fa", "on", "-ctk", "f16", "-ctv", "f16", "--jinja",
            "--no-webui", "--slot-save-path", SLOTDIR]
    env = dict(os.environ, **ENV_ADD)
    lf = open(f"{R}/server-{arm}.log", "w")
    p = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError(f"server {arm} exited rc={p.returncode}")
        try:
            if get("/health").get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        raise RuntimeError("server load timeout")
    env_rb = open(f"/proc/{p.pid}/environ", "rb").read().split(b"\0")
    env_rb = sorted(e.decode() for e in env_rb if e.startswith((b"GGML_", b"OMP_")))
    numa = subprocess.run(["bash", "-c", f"grep -c interleave:0-3 /proc/{p.pid}/numa_maps"],
                          capture_output=True, text=True).stdout.strip()
    log({"kind": "server_up", "arm": arm, "t": t, "tb": tb, "pid": p.pid,
         "load_s": round(time.time() - t0, 1), "argv": argv, "env_readback": env_rb,
         "numa_interleave_mappings": numa})
    return p


def stop(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(30)
    assert p.poll() is not None


def completion(prompt, n_predict):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "cache_prompt": True, "stream": False, "samplers": ["top_k"]}
    t0 = time.time()
    r = post("/completion", body)
    r["_wall_s"] = time.time() - t0
    return r


def restore(d):
    r = post("/slots/0?action=restore", {"filename": f"d{d}.bin"})
    return r.get("n_restored"), r.get("timings", {}).get("restore_ms")


def main():
    os.makedirs(SLOTDIR, exist_ok=True)
    tok = None
    for arm, t, tb in ARMS:
        p = start(arm, t, tb)
        try:
            if tok is None:
                tok_path = f"{R}/corpus_tokens.json"
                if os.path.exists(tok_path):
                    tok, probe = json.load(open(tok_path))
                else:
                    text = open(CORPUS).read()
                    tok = post("/tokenize", {"content": text, "add_special": False})["tokens"]
                    probe = post("/tokenize", {"content": "\n\nQuestion: Summarize the preceding "
                                 "text in detail, section by section.\nAnswer:", "add_special": False})["tokens"]
                    json.dump([tok, probe], open(tok_path, "w"))
                log({"kind": "corpus", "tokens": len(tok), "probe_tokens": len(probe)})
                assert len(tok) >= DEPTHS[-1] + TAIL
            if not all(os.path.exists(f"{SLOTDIR}/d{d}.bin") for d in DEPTHS):
                post("/slots/0?action=erase", {})
                for d in DEPTHS:
                    r = completion(tok[:d], 1)
                    tm = r["timings"]
                    s = post("/slots/0?action=save", {"filename": f"d{d}.bin"})
                    log({"kind": "grow", "arm": arm, "t": t, "tb": tb, "depth": d,
                         "prompt_n": tm["prompt_n"], "prompt_ms": tm["prompt_ms"],
                         "prompt_tps": tm["prompt_per_second"], "saved_tokens": s.get("n_saved"),
                         "save_ms": s.get("timings", {}).get("save_ms")})
            for d in DEPTHS:
                n_res, res_ms = restore(d)
                r = completion(tok[:d + TAIL], 1)
                tm = r["timings"]
                log({"kind": "prefill_at_depth", "arm": arm, "t": t, "tb": tb, "depth": d,
                     "restored": n_res, "restore_ms": res_ms, "prompt_n": tm["prompt_n"],
                     "prompt_tps": tm["prompt_per_second"], "reprefill": tm["prompt_n"] > TAIL + 8})
                for rep in range(B_REPS):
                    n_res, res_ms = restore(d)
                    r = completion(tok[:d] + probe, N_PRED)
                    tm = r["timings"]
                    log({"kind": "decode_at_depth", "arm": arm, "t": t, "tb": tb, "depth": d,
                         "rep": rep, "restored": n_res, "prompt_n": tm["prompt_n"],
                         "predicted_n": tm["predicted_n"], "pred_tps": tm["predicted_per_second"],
                         "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
                         "reprefill": tm["prompt_n"] > len(probe) + 8,
                         "content_sha": hashlib.sha256(r["content"].encode()).hexdigest()[:16],
                         "content": r["content"]})
        finally:
            stop(p)
            log({"kind": "server_down", "arm": arm})


if __name__ == "__main__":
    main()
