#!/usr/bin/env python3
"""SPQ-02/03: D1 gate-G1 language instrument, CASCADE arm (CPU correctness; no stack, no server).

  arm tts_rt : FLEURS test sentences (en_us, it_it) -> frozen qwen-tts (production-speech-v1, CPU, one thread per
               process) -> 24k wav -> 16k -> frozen whisper-cli large-v3-turbo (language forced AND auto) -> WER/CER.
               This is the cascade's *speech OUTPUT* language gate (the D2 gate G1) and the baseline any candidate
               interlocutor's wavs are scored against (--score-dir).
  arm stt    : real FLEURS human audio (en_us, it_it) -> whisper-cli -> WER/CER. Cascade STT baseline per language.

TTS text preparation (CS-8.L, operator listen check 2026-10-22): the TTS input is passed through the orchestrator's
language-gated normalizer (epyc-orchestrator src/voice/tts_text_normalize.py: Italian numbers, units, letter acronyms;
other languages unchanged) BEFORE qwen-tts. Default ON for this eval path; --no-tts-normalize feeds the raw FLEURS text
(byte-identical to the SPQ-03 run of 2026-10-21, which predates the flag: its config_echo has no tts_text_normalize key).
The ASR reference (ref_norm) is always the raw FLEURS text. The normalizer module's sha256 is stamped in config_echo.json.
Tracked copy of /mnt/raid0/llm/tmp/speech-queue-20261020/scripts/g1_cascade_language.py (sha256 c362894b..., the SPQ-03
run script, left untouched as that run's provenance); the only change is this text-preparation flag.

Candidates (Step-Audio 2 mini, Qwen3-Omni) plug in by writing <prompt_id>.wav files for the SAME prompts
(`prompts.json` is written, sha256-stamped) and running:  --arms score --score-dir DIR --label step2mini

Resource class: CPU CORRECTNESS. Self-claims a build-role region claim (region-lock) unless --no-claim.
Never contacts a server or any production port. Every child PID is captured; none outlives the script.

usage: g1_cascade_language.py --deadline ISO-UTC [--n 60] [--regions q0] [--procs 20] [--arms tts_rt,stt] [--dry-run]
                              [--no-tts-normalize] [--tts-normalizer PATH]
"""
import argparse, datetime, hashlib, importlib.util, json, os, re, signal, subprocess, sys, tarfile, time, unicodedata, wave
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/mnt/raid0/llm"
OUT_DEFAULT = f"{ROOT}/tmp/speech-queue-20261020/g1_out"
FLEURS = f"{ROOT}/data/speech-testsets-20261020/fleurs/data"
QWEN = f"{ROOT}/qwentts.cpp/build"
WHBIN = f"{ROOT}/whisper.cpp/build/bin"
TTS_MODEL = f"{ROOT}/models/Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf"
TTS_CODEC = f"{ROOT}/models/Qwen3-TTS-qwentts/qwen-tokenizer-12hz-Q8_0.gguf"
WH_MODEL = f"{ROOT}/models/whisper-ggml/ggml-large-v3-turbo.bin"
SHIM = f"{ROOT}/cache/shims/nprocs_shim.so"
LINKAGE = "/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
RL = "/workspace/repos/epyc-orchestrator/scripts/region-lock"
PY = "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python"
TTS_NORMALIZER = "/workspace/repos/epyc-orchestrator/src/voice/tts_text_normalize.py"
LANGS = {"en": ("en_us", "English", "en"), "it": ("it_it", "Italian", "it")}
CHILDREN = set()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def load_tts_normalizer(path):
    """Load normalize_for_tts(text, language) from the orchestrator module file (stdlib-only, no package import)."""
    spec = importlib.util.spec_from_file_location("tts_text_normalize", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_for_tts


def tts_text(text, lg, normalizer):
    """The text handed to qwen-tts: raw when normalization is off (normalizer None), else language-gated normalization."""
    return text if normalizer is None else normalizer(text, lg)


def norm(t, lang):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[^\w\s']", " ", t.replace("’", "'"))
    t = re.sub(r"\b(\d+)\b", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def select_prompts(n):
    """Deterministic: unique FLEURS test sentences, 8..24 words, every k-th by sorted id."""
    P = {}
    for lg, (fl, _, _) in LANGS.items():
        seen, rows = set(), []
        for ln in open(f"{FLEURS}/{fl}/test.tsv", encoding="utf-8"):
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4 or c[0] in seen:
                continue
            w = len(c[2].split())
            if 8 <= w <= 24:
                seen.add(c[0]); rows.append((int(c[0]), c[1], c[2], c[3]))
        rows.sort()
        step = max(1, len(rows) // n)
        P[lg] = [dict(id=f"{lg}-{r[0]}", sid=r[0], audio=r[1], text=r[2], ref_norm=norm(r[2], lg)) for r in rows[::step][:n]]
    return P


def run_child(argv, env, stdin=None, outp=None, timeout=600):
    f = open(outp, "wb") if outp else subprocess.DEVNULL
    p = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=f,
                         stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.add(p.pid)
    try:
        p.communicate(stdin.encode() if stdin is not None else None, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait()
        return -9
    finally:
        CHILDREN.discard(p.pid)
        if outp: f.close()
    return p.returncode


def resample_16k(src, dst):
    import numpy as np
    from scipy.signal import resample_poly
    import struct
    raw = open(src, "rb").read()          # RIFF parse: the stdlib wave module rejects IEEE-float (format 3) FLEURS wavs
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "not a RIFF/WAVE file"
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid, csz = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + csz]
        if cid == b"fmt ": fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data": data = body
        pos += 8 + csz + (csz & 1)
    tag, ch, sr, _, _, bits = fmt
    if tag == 0xFFFE: tag = 3 if bits == 32 else 1   # WAVE_FORMAT_EXTENSIBLE: sample type inferred from width
    if tag == 1 and bits == 16: x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif tag == 3 and bits == 32: x = np.frombuffer(data[:len(data) // 4 * 4], dtype=np.float32) * 32768.0
    else: raise ValueError(f"unsupported wav format tag={tag} bits={bits}: {src}")
    x = x[:len(x) // ch * ch].reshape(-1, ch).mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g)
    y = np.clip(x, -32768, 32767).astype(np.int16)
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(y.tobytes())
    return len(y) / 16000.0


def whisper(wav16, lang, outbase, cpu):
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    env.update({"HIP_VISIBLE_DEVICES": "-1", "LD_LIBRARY_PATH": f"{WHBIN}:/opt/rocm/lib"})
    rc = run_child(["taskset", "-c", str(cpu), f"{WHBIN}/whisper-cli", "-m", WH_MODEL, "-t", "1", "-ng", "-l", lang, "-nt",
                    "-otxt", "-of", outbase, "-f", wav16], env, timeout=900)
    try:
        return rc, open(outbase + ".txt", encoding="utf-8").read().strip()
    except OSError:
        return rc, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", required=True)
    ap.add_argument("--n", type=int, default=60, help="sentences per language")
    ap.add_argument("--regions", default="q0")
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--arms", default="tts_rt,stt")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--score-dir", default=None)
    ap.add_argument("--label", default="cascade_qwentts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-claim", action="store_true")
    ap.add_argument("--claim-wait", type=int, default=900)
    ap.add_argument("--cpus", default=None, help="cpu list for the children (set by the claim wrapper)")
    ap.add_argument("--tts-normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="normalize the TTS input text (default on; --no-tts-normalize = raw FLEURS text)")
    ap.add_argument("--tts-normalizer", default=TTS_NORMALIZER, help="path of the orchestrator tts_text_normalize.py")
    a = ap.parse_args()
    dl = datetime.datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
    if now() >= dl:
        print("deadline already passed", file=sys.stderr); return 3
    dl_hard = dl
    dl = dl - datetime.timedelta(seconds=90)  # stop taking new work 90 s early so results.json is always written before the hard timeout
    arms = a.arms.split(",")
    # ---- preflight (pure file checks) ----
    need = [WH_MODEL, f"{WHBIN}/whisper-cli", FLEURS + "/it_it/test.tsv", FLEURS + "/en_us/test.tsv",
            FLEURS + "/it_it/audio/test.tar.gz", FLEURS + "/en_us/audio/test.tar.gz", RL, PY]
    if "tts_rt" in arms:
        need += [f"{QWEN}/qwen-tts", TTS_MODEL, TTS_CODEC, SHIM, LINKAGE]
        if a.tts_normalize:
            need.append(a.tts_normalizer)
    miss = [p for p in need if not os.path.exists(p)]
    prompts = select_prompts(a.n) if not any(m.endswith("test.tsv") for m in miss) else None
    plan = dict(arms=arms, tts_text_normalize=bool(a.tts_normalize and "tts_rt" in arms), n_per_lang=a.n, procs=a.procs, regions=a.regions, deadline=a.deadline, out=a.out, missing=miss,
                prompts={k: len(v) for k, v in (prompts or {}).items()}, est_runtime_min="10-25")
    if a.dry_run:
        print(json.dumps(plan, indent=1)); return 1 if miss else 0
    if miss:
        print("MISSING", miss, file=sys.stderr); return 2
    normalizer = load_tts_normalizer(a.tts_normalizer) if plan["tts_text_normalize"] else None
    # ---- claim wrapper (build role: correctness work, regions not held by others) ----
    if not a.no_claim and os.environ.get("G1_INNER") != "1":
        os.environ["G1_INNER"] = "1"
        left = int((dl_hard - now()).total_seconds())
        os.execv(RL, [RL, "run", "--regions", a.regions, "--role", "build", "--timeout-s", str(a.claim_wait), "--tag", "spq-g1-language", "--",
                      "timeout", "--signal=TERM", str(left), PY, "-I", os.path.abspath(__file__)] + sys.argv[1:])
    cpus = []
    cl = a.cpus or open("/proc/self/status").read().split("Cpus_allowed_list:")[1].split("\n")[0].strip()
    for part in cl.split(","):
        lo, _, hi = part.partition("-"); cpus += list(range(int(lo), int(hi or lo) + 1))
    cpus = cpus[:a.procs]
    O = a.out; os.makedirs(O + "/wav", exist_ok=True); os.makedirs(O + "/asr", exist_ok=True); os.makedirs(O + "/real16", exist_ok=True)
    json.dump(prompts, open(O + "/prompts.json", "w"), ensure_ascii=False, indent=1)
    cfg = dict(plan, utc=now().isoformat(), cpus_used=cpus, prompts_sha256=sha256_file(O + "/prompts.json"),
               shas={os.path.basename(p): sha256_file(p) for p in need if os.path.isfile(p) and not p.endswith(".tar.gz")},
               tts_normalizer=dict(path=a.tts_normalizer, sha256=sha256_file(a.tts_normalizer)) if normalizer else None,
               qwen_cmd="qwen-tts --model M --codec C --lang <Language> --seed 1234 --ref-wav REF -o OUT (stdin=text); SHIM_NPROCS=2 (1 thread)")
    json.dump(cfg, open(O + "/config_echo.json", "w"), indent=1)
    if "tts_rt" in arms:
        e = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
        e.update({"HIP_VISIBLE_DEVICES": "-1", "GGML_BACKEND": "CPU", "LD_LIBRARY_PATH": f"{QWEN}:/opt/rocm/lib"})
        rc = run_child([LINKAGE, f"{QWEN}/qwen-tts", QWEN], e, outp=O + "/linkage_qwen-tts.txt", timeout=120)
        if rc != 0:
            print("LINKAGE FAIL", file=sys.stderr); return 4
    # extract needed real-speech members (reference clips + stt arm)
    refs = {}
    for lg, (fl, _, _) in LANGS.items():
        want = {p["audio"] for p in prompts[lg]}
        with tarfile.open(f"{FLEURS}/{fl}/audio/test.tar.gz") as t:
            for m in t:
                b = os.path.basename(m.name)
                if not (m.isfile() and b.endswith(".wav")):
                    continue
                if b in want:
                    t.extract(m, O + f"/real_{lg}", filter="data")
                elif lg not in refs:  # first clip outside the prompt set = voice-clone reference for the Base model
                    t.extract(m, O + f"/real_{lg}", filter="data"); refs[lg] = O + f"/real_{lg}/" + m.name
    rows = []
    jobs = [(lg, p) for lg in LANGS for p in prompts[lg]]

    def cpu_for(i): return cpus[i % len(cpus)]

    def do_tts(i, lg, p):
        if now() >= dl: return None
        raw, w16 = f"{O}/wav/{a.label}_{p['id']}.24k.wav", f"{O}/wav/{p['id']}.wav"
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
        env.update({"HIP_VISIBLE_DEVICES": "-1", "GGML_BACKEND": "CPU", "LD_LIBRARY_PATH": f"{QWEN}:/opt/rocm/lib", "SHIM_NPROCS": "2", "LD_PRELOAD": SHIM})
        text = tts_text(p["text"], lg, normalizer)
        t0 = time.time()
        rc = run_child(["taskset", "-c", str(cpu_for(i)), f"{QWEN}/qwen-tts", "--model", TTS_MODEL, "--codec", TTS_CODEC, "--lang", LANGS[lg][1],
                        "--seed", "1234", "--ref-wav", refs[lg], "-o", raw], env, stdin=text + "\n", outp=f"{O}/wav/{p['id']}.tts.log", timeout=900)
        dt = time.time() - t0
        if rc != 0 or not os.path.exists(raw): return dict(id=p["id"], lang=lg, arm="tts_rt", tts_rc=rc, wer=None, failed="tts")
        dur = resample_16k(raw, w16)
        return dict(id=p["id"], lang=lg, arm="tts_rt", tts_s=dt, audio_s=dur, wav=w16, tts_text=text, tts_text_changed=text != p["text"])

    def do_asr(i, lg, p, wav, arm, extra):
        if now() >= dl: return None
        res = dict(extra)
        for mode, lflag in (("forced", LANGS[lg][2]), ("auto", "auto")):
            rc, hyp = whisper(wav, lflag, f"{O}/asr/{arm}_{p['id']}_{mode}", cpu_for(i))
            h, r = norm(hyp, lg).split(), p["ref_norm"].split()
            res[mode] = dict(rc=rc, hyp=hyp, wer=(edit(r, h) / max(1, len(r))), cer=(edit(list(p["ref_norm"]), list(norm(hyp, lg))) / max(1, len(p["ref_norm"]))),
                             ref_words=len(r))
        return res

    stage1 = []
    if "tts_rt" in arms:
        with ThreadPoolExecutor(len(cpus)) as ex:
            futs = [ex.submit(do_tts, i, lg, p) for i, (lg, p) in enumerate(jobs)]
            stage1 = [f.result() for f in futs]
    for r in stage1:
        if r and r.get("wav"):
            lg = r["lang"]; p = next(x for x in prompts[lg] if x["id"] == r["id"])
            r["_p"] = p
    work = []
    for i, r in enumerate([r for r in stage1 if r and r.get("wav")]):
        work.append((i, r["lang"], r["_p"], r["wav"], "tts_rt", {k: v for k, v in r.items() if k != "_p"}))
    if "score" in arms and a.score_dir:
        for i, (lg, p) in enumerate(jobs):
            w = os.path.join(a.score_dir, p["id"] + ".wav")
            if os.path.exists(w):
                w16 = f"{O}/wav/{a.label}_{p['id']}.16k.wav"; d = resample_16k(w, w16)
                work.append((i, lg, p, w16, "score", dict(id=p["id"], lang=lg, arm=a.label, audio_s=d, wav=w16)))
    if "stt" in arms:
        for i, (lg, p) in enumerate(jobs):
            src = None
            for root, _, fs in os.walk(O + f"/real_{lg}"):
                if p["audio"] in fs: src = os.path.join(root, p["audio"])
            if src:
                w16 = f"{O}/real16/{p['id']}.wav"; d = resample_16k(src, w16)
                work.append((i, lg, p, w16, "stt", dict(id=p["id"], lang=lg, arm="stt", audio_s=d)))
    with ThreadPoolExecutor(len(cpus)) as ex, open(O + "/items.jsonl", "w") as jl:
        futs = [ex.submit(do_asr, *w) for w in work]
        for f in as_completed(futs):
            r = f.result()
            if r: rows.append(r); jl.write(json.dumps(r, ensure_ascii=False) + "\n"); jl.flush()
    summ = {}
    for r in rows:
        for mode in ("forced", "auto"):
            s = summ.setdefault(f"{r['arm']}|{r['lang']}|{mode}", dict(n=0, wer_sum=0.0, cer_sum=0.0, words=0, errs=0.0))
            s["n"] += 1; s["wer_sum"] += r[mode]["wer"]; s["cer_sum"] += r[mode]["cer"]
            s["words"] += r[mode]["ref_words"]; s["errs"] += r[mode]["wer"] * r[mode]["ref_words"]
    for k, s in summ.items():
        s["corpus_wer"] = s["errs"] / max(1, s["words"]); s["mean_cer"] = s["cer_sum"] / s["n"]
    json.dump(dict(config=cfg, summary=summ, n_items=len(rows), finished_utc=now().isoformat(),
                   note="screening numbers: pinned simple normaliser, NOT the Annex S P-STT-1 pipeline; Italian intelligibility needs the human listen check (wav/ dir)"),
              open(O + "/results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        for pid in list(CHILDREN):
            try: os.killpg(pid, signal.SIGKILL)
            except Exception: pass
    sys.exit(rc)
