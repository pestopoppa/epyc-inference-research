#!/usr/bin/env python3
"""VL quality comparison on the MI210 — Qwen3-VL family vs the incumbent vs MiniCPM-o.

Why this exists: the only paired quality evidence on record is n=10 with ONE
discordant pair (McNemar p=1.0), and that single pair turned out to be a scoring
artifact — the incumbent answered "0.11 kWh" where the accepted answer was "0.11".
Corrected, the two models tied 7/10. Meanwhile Qwen3-VL 4B/8B/30B-A3B have been on
disk unevaluated, a full generation newer than the deployed Qwen2.5-VL-7B.

Each arm runs at the BEST quant present on disk, not a lowest-common-denominator
quant. 4B/8B and MiniCPM-o run Q8_0; the incumbent Qwen2.5-VL-7B and the
30B-A3B exist only as Q4_K_M. Forcing everything to Q4_K_M would have meant
running the newer models below their available fidelity to buy a uniformity that
nothing here actually requires. The quant column is reported on every row so the
mismatch stays visible: if a Q8_0 arm beats the Q4_K_M incumbent narrowly, quant
is a live confound and the follow-up is to quantize, not to conclude.

Scoring records BOTH:
  * strict  — the suite's own exact_match/substring rule, as the harness does today.
  * lenient — numeric-and-unit tolerant: strips units, %, commas, currency, and
    compares numerically when both sides parse as numbers.
The gap between them IS the scoring-artifact rate. Reporting only strict is what
produced the phantom +10pp, so this harness refuses to report one without the other.

Per-question results are persisted so a paired (McNemar) comparison can be run
across any two arms afterwards without re-inferencing.
"""
import base64, json, re, subprocess, sys, time, urllib.request, os, signal

GPU_D = "/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip/bin"
M = "/mnt/raid0/llm/models"
LC = f"{M}/lmstudio-community"
OUT = "/mnt/raid0/llm/tmp/vlquality_results.json"
PORT = 19870

ARMS = [
    dict(key="qwen25vl_7b",   label="Qwen2.5-VL-7B Q4_K_M (INCUMBENT)",
         model=f"{LC}/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
         mmproj=f"{LC}/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf", extra=[]),
    dict(key="qwen3vl_4b",    label="Qwen3-VL-4B Q8_0",
         model=f"{LC}/Qwen3-VL-4B-Instruct-GGUF/Qwen3-VL-4B-Instruct-Q8_0.gguf",
         mmproj=f"{LC}/Qwen3-VL-4B-Instruct-GGUF/mmproj-Qwen3-VL-4B-Instruct-F16.gguf", extra=[]),
    dict(key="qwen3vl_8b",    label="Qwen3-VL-8B Q8_0",
         model=f"{LC}/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf",
         mmproj=f"{LC}/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3-VL-8B-Instruct-F16.gguf", extra=[]),
    dict(key="qwen3vl_30b",   label="Qwen3-VL-30B-A3B Q4_K_M",
         model=f"{LC}/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
         mmproj=f"{LC}/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf", extra=[]),
    dict(key="minicpm_o",     label="MiniCPM-o-4.5 Q8_0",
         model=f"{M}/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf",
         mmproj=f"{M}/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf", extra=[]),
]

NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def norm_strict(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def norm_lenient(s):
    """Strip the decorations that turn a right answer into a scored miss."""
    s = (s or "").strip().lower()
    s = re.sub(r"[$€£%]", "", s)
    s = re.sub(r"\b(kwh|kg|mg|km|cm|mm|ml|g|m|s|usd|eur|gbp|units?|people|years?)\b", "", s)
    s = s.replace(",", "").replace("$", "")
    return re.sub(r"\s+", " ", s).strip(" .:")


def as_num(s):
    m = NUM.findall(s or "")
    if len(m) != 1:
        return None
    try:
        return float(m[0].replace(",", "."))
    except ValueError:
        return None


def score(expected, got, method, cfg):
    e_s, g_s = norm_strict(expected), norm_strict(got)
    if method == "substring":
        strict = e_s in g_s
    else:
        strict = e_s == g_s
    e_l, g_l = norm_lenient(expected), norm_lenient(got)
    lenient = (e_l == g_l) or (e_l in g_l and len(e_l) > 0)
    en, gn = as_num(e_l), as_num(g_l)
    if en is not None and gn is not None:
        lenient = lenient or abs(en - gn) < 1e-9
    return bool(strict), bool(strict or lenient)


def wait_ready(log, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if "model loaded" in open(log, errors="ignore").read():
                return True
        except FileNotFoundError:
            pass
        time.sleep(3)
    return False


def ask(img_b64, prompt, nothink):
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
        {"type": "text", "text": prompt + "\nReply with only the answer, no explanation."}]}],
        "max_tokens": 64, "temperature": 0.0, "seed": 42}
    if nothink:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["enable_thinking"] = False
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return (msg.get("content") or "").strip(), d.get("usage", {})


def main():
    import yaml
    suite = yaml.safe_load(open(
        "/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/debug/vl.yaml"))
    qs = suite["questions"]
    print(f"suite: {len(qs)} questions", flush=True)

    imgs = {}
    for q in qs:
        p = q["image_path"]
        if p not in imgs:
            imgs[p] = base64.b64encode(open(p, "rb").read()).decode()

    results = {}
    for arm in ARMS:
        if not os.path.exists(arm["model"]) or not os.path.exists(arm["mmproj"]):
            print(f"SKIP {arm['key']}: weights missing", flush=True)
            continue
        log = f"/mnt/raid0/llm/tmp/vq_{arm['key']}.log"
        open(log, "w").close()
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{GPU_D}:/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
        cmd = ["taskset", "-c", "184-191", f"{GPU_D}/llama-server",
               "-m", arm["model"], "--mmproj", arm["mmproj"],
               "--host", "127.0.0.1", "--port", str(PORT),
               "-np", "1", "-c", "16384", "-t", "8", "-ngl", "999",
               "--device", "ROCm0", "--jinja", "--log-colors", "off"] + arm["extra"]
        srv = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_ready(log):
                print(f"FAIL {arm['key']}: never loaded", flush=True)
                results[arm["key"]] = {"label": arm["label"], "error": "never loaded"}
                continue
            # MiniCPM-o is a reasoning model: without thinking off it emits
            # reasoning_content and an EMPTY content, scoring 0. Verified today.
            nothink = arm["key"] == "minicpm_o"
            rows, t0 = [], time.time()
            for i, q in enumerate(qs):
                try:
                    got, usage = ask(imgs[q["image_path"]], q["prompt"], nothink)
                except Exception as e:
                    got, usage = f"<ERROR {e}>", {}
                st, le = score(q["expected"], got, q.get("scoring_method", "exact_match"),
                               q.get("scoring_config", {}))
                rows.append(dict(id=q["id"], dataset=q.get("source_dataset"),
                                 expected=q["expected"], got=got, strict=st, lenient=le,
                                 prompt_tokens=usage.get("prompt_tokens")))
                if (i + 1) % 10 == 0:
                    print(f"  {arm['key']} {i+1}/{len(qs)}", flush=True)
            el = time.time() - t0
            s = sum(r["strict"] for r in rows)
            l = sum(r["lenient"] for r in rows)
            med_pt = sorted(r["prompt_tokens"] or 0 for r in rows)[len(rows) // 2]
            results[arm["key"]] = dict(label=arm["label"], n=len(rows), strict=s, lenient=l,
                                       elapsed_s=round(el, 1), median_prompt_tokens=med_pt,
                                       rows=rows)
            print(f"== {arm['label']}: strict {s}/{len(rows)}  lenient {l}/{len(rows)}  "
                  f"({el:.0f}s, median image+text tokens {med_pt})", flush=True)
        finally:
            srv.send_signal(signal.SIGTERM)
            try:
                srv.wait(timeout=25)
            except subprocess.TimeoutExpired:
                srv.kill()
            time.sleep(5)
        json.dump(results, open(OUT, "w"), indent=1)

    json.dump(results, open(OUT, "w"), indent=1)
    print("\n=== SUMMARY (all arms Q4_K_M, MI210) ===", flush=True)
    print(f"{'arm':<40} {'strict':>8} {'lenient':>8} {'artifact':>9} {'img tok':>8}")
    for k, v in results.items():
        if "error" in v:
            print(f"{v['label']:<40} {'ERROR':>8}")
            continue
        print(f"{v['label']:<40} {v['strict']:>3}/{v['n']:<4} {v['lenient']:>3}/{v['n']:<4} "
              f"{v['lenient']-v['strict']:>9} {v['median_prompt_tokens']:>8}")
    print("=== VLQUALITY DONE ===", flush=True)


if __name__ == "__main__":
    main()
