#!/usr/bin/env python3
"""Final vision-model selection bench on the MI210 — MMMU validation (multiple choice).

Adapted from /mnt/raid0/llm/tmp/vlquality.py (same server launch, same taskset,
same one-arm-at-a-time discipline). What changed and why:

  * SUITE. vlquality.py ran OCRBench+ChartQA, the saturated family where
    Qwen3-VL was never claimed to beat Qwen2.5-VL (published deltas +0.4 to
    +8). That run could not separate the candidates and did not. This runs
    MMMU validation, the axis with the published +11.0 delta.
  * SCORING. Multiple choice -> extract a letter, compare exactly. No
    substring/lenient rule, because there is nothing to be lenient about.
    Parse failures are counted SEPARATELY per arm and reported: a cross-arm
    parse-fail gap is a scorer bug, not a capability gap (this project has
    been burned by exactly that).
  * THINKING. If an arm returns empty content while emitting reasoning_content,
    the harness flips that arm to enable_thinking=false and retries, so a
    reasoning model cannot silently score 0.

Every raw response is persisted so scoring can be redone offline.
"""
import base64, json, os, re, signal, subprocess, sys, time, urllib.error, urllib.request

GPU_D = "/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip/bin"
LC = "/mnt/raid0/llm/models/lmstudio-community"
MANIFEST = "/mnt/raid0/llm/tmp/mmmu_manifest.json"
OUT = "/mnt/raid0/llm/tmp/vision_final_results.json"
PORT = 19871

ARMS = [
    dict(key="qwen25vl_7b", label="Qwen2.5-VL-7B-Instruct", quant="Q4_K_M",
         model=f"{LC}/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
         mmproj=f"{LC}/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"),
    dict(key="qwen3vl_4b", label="Qwen3-VL-4B-Instruct", quant="Q8_0",
         model=f"{LC}/Qwen3-VL-4B-Instruct-GGUF/Qwen3-VL-4B-Instruct-Q8_0.gguf",
         mmproj=f"{LC}/Qwen3-VL-4B-Instruct-GGUF/mmproj-Qwen3-VL-4B-Instruct-F16.gguf"),
    dict(key="qwen3vl_8b", label="Qwen3-VL-8B-Instruct", quant="Q8_0",
         model=f"{LC}/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf",
         mmproj=f"{LC}/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3-VL-8B-Instruct-F16.gguf"),
    dict(key="qwen3vl_30b", label="Qwen3-VL-30B-A3B-Instruct", quant="Q4_K_M",
         model=f"{LC}/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
         mmproj=f"{LC}/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"),
]

LETTERS = "ABCDEFGHIJ"
INSTR = "Answer with the option's letter from the given choices directly."


def build_prompt(q):
    body = re.sub(r"<image\s*1\s*>", "", q["question"]).strip()
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return f"{body}\n\n{opts}\n\n{INSTR}"


def norm(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9.\-/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def _letter_rules(t, valid, tag):
    """Explicit-form rules, applied to one candidate slice of the response."""
    t = t.strip()
    t = re.sub(r"^\**\s*|\s*\**$", "", t)            # strip markdown bold
    t = re.sub(r"^(?:answer|final answer)\s*[:\-]\s*", "", t, flags=re.I)
    if not t:
        return None, None
    # \boxed{C} — models that finish a derivation in LaTeX
    m = re.search(r"\\boxed\{\s*\(?\s*([A-Za-z])\s*\)?[\.\s\}]", t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "boxed" + tag
    # the whole slice IS a letter: "A", "(A)", "A.", "**A**"
    m = re.fullmatch(r"[\(\[\{]?\s*([A-Za-z])\s*[\)\]\}\.\,:]?", t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "bare" + tag
    # "the answer is A" / "answer: (A)" / "option A" — take the LAST such
    ms = re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-Za-z])\b", t, re.I)
    ms = [x for x in ms if x.upper() in valid]
    if ms:
        return ms[-1].upper(), "phrase" + tag
    # leading letter then a delimiter: "A) foo", "A. foo", "A - foo"
    m = re.match(r"[\(\[]?\s*([A-Za-z])\s*[\)\]\.\:\,\-]\s+", t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "leading" + tag
    return None, None


def extract_letter(text, options):
    """Return (letter, method) or (None, 'parse_fail').

    Applied to the whole reply first, then to the TAIL (last line, last 300
    chars). The tail matters: models that ignore 'answer directly' and reason
    first put the letter at the END, and a whole-string-only parser scores
    those as failures -- which is a harness artifact, not a capability gap.
    """
    valid = set(LETTERS[:len(options)])
    if not text or not text.strip():
        return None, "empty"
    full = text.strip()
    lines = [l for l in full.splitlines() if l.strip()]
    slices = [(full, ""), (lines[-1] if lines else "", "_last"), (full[-300:], "_tail")]
    for s, tag in slices:
        if not s:
            continue
        got, meth = _letter_rules(s, valid, tag)
        if got:
            return got, meth
    # exact/normalised match against an option's text
    nt = norm(full)
    if nt:
        exact = [i for i, o in enumerate(options) if norm(o) == nt]
        if len(exact) == 1:
            return LETTERS[exact[0]], "option_text"
        contains = [i for i, o in enumerate(options) if norm(o) and norm(o) in nt]
        if len(contains) == 1:
            return LETTERS[contains[0]], "option_substr"
    # last resort: exactly one distinct standalone capital letter anywhere
    cands = [c for c in re.findall(r"\b([A-J])\b", full) if c in valid]
    if cands and len(set(cands)) == 1:
        return cands[0], "loose"
    return None, "parse_fail"


def wait_ready(log, proc, timeout=600, want=1):
    """Wait for the `want`-th 'model loaded'. Counting matters: the log is
    appended across restarts, so a substring test would return True instantly
    for restart #2 against restart #1's line and we would hammer a dead port."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if open(log, errors="ignore").read().count("model loaded") >= want:
                return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    return False


def vram_mb():
    try:
        o = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                           capture_output=True, text=True, timeout=30).stdout
        d = json.loads(o)
        for v in d.values():
            for k, val in v.items():
                if "Used" in k and "VRAM" in k:
                    return round(int(val) / 1048576)
    except Exception:
        pass
    try:
        o = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                           capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", o)
        if m:
            return round(int(m.group(1)) / 1048576)
    except Exception:
        pass
    return None


def ask(img_b64, prompt, nothink):
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
        {"type": "text", "text": prompt}]}],
        # temperature 0.2 = the production vision-role serving temperature.
        # Project rule: sampling-sensitive benches run at PRODUCTION temp with a
        # fixed seed, never greedy. (vlquality.py's temperature 0.0 was off-recipe.)
        # max_tokens 512, not 128: at 128 EVERY parse failure sat exactly at the
        # cap -- 3 for the incumbent but 41 for the 4B and 50 for the 8B, i.e.
        # the cap was silently penalising the models that reason before
        # answering, by up to 20% of the suite. That is a harness artifact and
        # it inverted the ranking.
        "max_tokens": 512, "temperature": 0.2, "seed": 42}
    if nothink:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return ((msg.get("content") or "").strip(),
            (msg.get("reasoning_content") or "").strip(),
            d.get("usage", {}))


def kill_and_verify(proc, label):
    if proc.poll() is not None:
        print(f"  [{label}] already exited rc={proc.returncode}", flush=True)
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] SIGTERM timed out -> SIGKILL", flush=True)
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    # CLAUDE.md process discipline: verify it is actually dead
    for _ in range(30):
        alive = subprocess.run(["ps", "-p", str(proc.pid)],
                               capture_output=True).returncode == 0
        if not alive:
            print(f"  [{label}] server pid {proc.pid} confirmed dead", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"server pid {proc.pid} for {label} would not die")


def main():
    qs = json.load(open(MANIFEST))
    print(f"MMMU sample: {len(qs)} multiple-choice questions", flush=True)
    imgs = {}
    for q in qs:
        p = q["image_path"]
        if p not in imgs:
            imgs[p] = base64.b64encode(open(p, "rb").read()).decode()

    only = sys.argv[1:] or None
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results.setdefault("_meta", {})
    results["_meta"].update(suite="MMMU validation (multiple-choice, single-image)",
                            n=len(qs), manifest=MANIFEST,
                            server=GPU_D, device="ROCm0 (MI210)")

    for arm in ARMS:
        if only and arm["key"] not in only:
            continue
        for f in (arm["model"], arm["mmproj"]):
            if not os.path.exists(f):
                print(f"SKIP {arm['key']}: missing {f}", flush=True)
                results[arm["key"]] = {"label": arm["label"], "error": f"missing {f}"}
                break
        else:
            log = f"/mnt/raid0/llm/tmp/vf_{arm['key']}.log"
            open(log, "w").close()
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = f"{GPU_D}:/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
            cmd = ["taskset", "-c", "184-191", f"{GPU_D}/llama-server",
                   "-m", arm["model"], "--mmproj", arm["mmproj"],
                   "--host", "127.0.0.1", "--port", str(PORT),
                   "-np", "1", "-c", "16384", "-t", "8", "-ngl", "999",
                   "--device", "ROCm0", "--jinja", "--log-colors", "off",
                   # Upstream warns at load: "Qwen-VL models require at minimum
                   # 1024 image tokens to function correctly ... try adding
                   # --image-min-tokens 1024". Applied to EVERY arm so the
                   # image-token floor is symmetric rather than whatever each
                   # model's baked-in default is. Probed working on all four
                   # arms (vision_flagprobe.json): 1108-1119 tokens on the same
                   # image, i.e. the floor lands identically across models.
                   "--image-min-tokens", "1024",
                   # Run 1 lost 92/250 questions on the incumbent when the
                   # server died mid-task at q158 with no error text. Every
                   # request here carries a DIFFERENT image, so the 8 GiB
                   # server prompt cache never hits -- the log was solid
                   # "making room for prompt cache entry, removing oldest
                   # entry". Pure churn, and the only subsystem doing large
                   # repeated host allocations. Disabled on every arm.
                   "--cache-ram", "0"]
            def boot():
                p = subprocess.Popen(cmd, stdout=open(log, "a"),
                                     stderr=subprocess.STDOUT, env=env)
                return p

            vram_before = vram_mb()
            t_load = time.time()
            srv = boot()
            restarts = [0]
            try:
                if not wait_ready(log, srv):
                    tail = open(log, errors="ignore").read()[-1500:]
                    print(f"FAIL {arm['key']}: never loaded\n{tail}", flush=True)
                    results[arm["key"]] = {"label": arm["label"], "quant": arm["quant"],
                                           "error": "never loaded", "log_tail": tail}
                    continue
                load_s = round(time.time() - t_load, 1)
                time.sleep(3)
                vram_after = vram_mb()
                vram_used = (vram_after - (vram_before or 0)) if vram_after else None
                print(f"-- {arm['label']} loaded in {load_s}s, VRAM {vram_after} MB "
                      f"(delta {vram_used} MB)", flush=True)

                nothink = False
                rows, t0 = [], time.time()
                for i, q in enumerate(qs):
                    prompt = build_prompt(q)
                    # Crash-resilient ask: run 1 lost 92 consecutive questions
                    # because the server died and every later request got
                    # ECONNREFUSED. A dead server must cost seconds, not an arm.
                    err = None
                    for attempt in range(4):
                        try:
                            got, think, usage = ask(imgs[q["image_path"]], prompt, nothink)
                            err = None
                            break
                        except Exception as e:
                            got, think, usage = "", "", {}
                            err = f"{type(e).__name__}: {e}"
                            dead = (srv.poll() is not None) or isinstance(
                                e, (urllib.error.URLError, ConnectionError))
                            if attempt == 3 or not dead:
                                break
                            restarts[0] += 1
                            print(f"  [{arm['key']}] server down at q{q['idx']} "
                                  f"({err}) -> restart #{restarts[0]}", flush=True)
                            kill_and_verify(srv, arm["key"] + "-crashed")
                            time.sleep(3)
                            srv = boot()
                            if not wait_ready(log, srv, timeout=300,
                                              want=restarts[0] + 1):
                                print(f"  [{arm['key']}] restart FAILED to load", flush=True)
                                break
                    # a reasoning model must not silently score 0
                    if not got and think and not nothink:
                        print(f"  [{arm['key']}] empty content + reasoning_content -> "
                              f"enabling enable_thinking=false for the rest of this arm",
                              flush=True)
                        nothink = True
                        try:
                            got, think, usage = ask(imgs[q["image_path"]], prompt, True)
                            err = None
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"
                    letter, method = extract_letter(got, q["options"])
                    rows.append(dict(idx=q["idx"], id=q["id"], subject=q["subject"],
                                     expected=q["answer"], raw=got,
                                     reasoning=think[:400] if think else None,
                                     pred=letter, parse=method,
                                     correct=(letter == q["answer"]),
                                     prompt_tokens=usage.get("prompt_tokens"),
                                     completion_tokens=usage.get("completion_tokens"),
                                     nothink=nothink, error=err))
                    if (i + 1) % 25 == 0:
                        c = sum(r["correct"] for r in rows)
                        print(f"  {arm['key']} {i+1}/{len(qs)}  running {c}/{i+1}", flush=True)
                el = time.time() - t0
                c = sum(r["correct"] for r in rows)
                pf = sum(r["parse"] == "parse_fail" for r in rows)
                em = sum(r["parse"] == "empty" for r in rows)
                errs = sum(bool(r["error"]) for r in rows)
                pts = sorted(r["prompt_tokens"] or 0 for r in rows)
                results[arm["key"]] = dict(
                    label=arm["label"], quant=arm["quant"], n=len(rows), correct=c,
                    pct=round(100 * c / len(rows), 1), parse_fail=pf, empty=em,
                    request_errors=errs, server_restarts=restarts[0],
                    elapsed_s=round(el, 1),
                    load_s=load_s, vram_mb_total=vram_after, vram_mb_model=vram_used,
                    median_prompt_tokens=pts[len(pts) // 2],
                    mean_prompt_tokens=round(sum(pts) / len(pts)),
                    used_nothink=nothink, rows=rows)
                print(f"== {arm['label']} {arm['quant']}: {c}/{len(rows)} "
                      f"({100*c/len(rows):.1f}%)  parse_fail={pf} empty={em} err={errs} "
                      f"restarts={restarts[0]}  "
                      f"{el:.0f}s  median img+txt tokens {pts[len(pts)//2]}", flush=True)
            finally:
                kill_and_verify(srv, arm["key"])
                time.sleep(5)
            json.dump(results, open(OUT, "w"), indent=1)

    json.dump(results, open(OUT, "w"), indent=1)
    print("\n=== SUMMARY: MMMU validation (multiple-choice, single-image), MI210 ===", flush=True)
    print(f"{'model':<28} {'quant':<8} {'score':>10} {'pct':>7} {'pfail':>6} "
          f"{'medtok':>7} {'wall_s':>8} {'VRAM_MB':>8}")
    for k, v in results.items():
        if k == "_meta":
            continue
        if "error" in v:
            print(f"{v['label']:<28} {v.get('quant',''):<8} {'ERROR: ' + v['error']}")
            continue
        print(f"{v['label']:<28} {v['quant']:<8} {v['correct']:>4}/{v['n']:<5} "
              f"{v['pct']:>6.1f}% {v['parse_fail']:>6} {v['median_prompt_tokens']:>7} "
              f"{v['elapsed_s']:>8.0f} {str(v['vram_mb_model']):>8}")
    print("=== VISION FINAL DONE ===", flush=True)


if __name__ == "__main__":
    main()
