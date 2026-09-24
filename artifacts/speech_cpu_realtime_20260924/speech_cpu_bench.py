#!/usr/bin/env python3
"""CPU real-time benchmark for production STT (whisper.cpp) and TTS (qwentts.cpp).

Runs the EXISTING production-speech-v1 binaries (HIP builds) in CPU mode:
  * HIP_VISIBLE_DEVICES=-1  -> ROCm sees zero devices, no VRAM is touched
  * whisper-server -ng      -> whisper CPU backend
  * GGML_BACKEND=CPU        -> qwentts forced onto CPU backend
  * LD_PRELOAD nprocs_shim  -> qwentts hardcodes threads = hardware_concurrency()/2
                               (ignores affinity: 96 on this host); the shim makes
                               get_nprocs() return SHIM_NPROCS = 2*T.
Every server is pinned with taskset to physical cores inside 0-79.
Only PIDs started here are ever signalled.

Usage: speech_cpu_bench.py <phase> [...]   (phases: stt_sweep, tts_sweep, concurrent)
"""
import http.client, json, os, signal, subprocess, sys, time, uuid, wave, threading

ART = "/mnt/raid0/llm/epyc-inference-research/artifacts/speech_cpu_realtime_20260924"
RAW = ART + "/raw"
TMP = "/mnt/raid0/llm/tmp/speech-cpu-20260924"
WH = "/mnt/raid0/llm/whisper.cpp"
QT = "/mnt/raid0/llm/qwentts.cpp"
SHORT_WAV = TMP + "/short_11s.wav"
LONG_WAV = TMP + "/long_85s.wav"
STT_PORT, TTS_PORT = 19100, 19102

TEXT_SHORT = "Hello, this is a short test of the text to speech system running on the processor."
TEXT_LONG = ("The history of computing is a story of steady compression. Machines that once filled "
             "entire rooms and consumed the power of a small town now fit in a pocket and run on a "
             "battery for a full day. Each generation of engineers inherited the constraints of the "
             "last and found new ways around them, trading one scarce resource for another. Memory "
             "was expensive, so programmers learned to be frugal with it. Processors were slow, so "
             "algorithms grew clever. Today the scarce resource is often attention itself, and the "
             "systems we build are judged less by how fast they compute than by how naturally they "
             "listen, understand, and respond.")


def wav_dur(p):
    w = wave.open(p)
    return w.getnframes() / w.getframerate()


def log(msg, fh=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def cores(lo, n):
    return f"{lo}-{lo + n - 1}"


# ---------------------------------------------------------------- servers
class Server:
    def __init__(self, name, argv, env_extra, corelist, port, logpath):
        self.name, self.port, self.logpath = name, port, logpath
        env = dict(os.environ)
        env.update(env_extra)
        self.argv = ["taskset", "-c", corelist] + argv
        self.env_extra = env_extra
        self.fh = open(logpath, "w")
        self.p = subprocess.Popen(self.argv, env=env, stdout=self.fh, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        self.pid = self.p.pid
        with open(RAW + "/pids_started.txt", "a") as f:
            f.write(f"{time.strftime('%F %T')} {name} pid={self.pid} argv={' '.join(self.argv)} env={env_extra}\n")

    def wait_ready(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError(f"{self.name} exited rc={self.p.returncode}; see {self.logpath}")
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                c.request("GET", "/health"); r = c.getresponse(); r.read()
                if r.status == 200:
                    return time.time() - t0
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"{self.name} not ready in {timeout}s")

    def stop(self):
        # only our own PID; TERM -> KILL; verify dead
        if self.p.poll() is None:
            os.kill(self.pid, signal.SIGTERM)
            try:
                self.p.wait(10)
            except subprocess.TimeoutExpired:
                os.kill(self.pid, signal.SIGKILL); self.p.wait(10)
        alive = os.path.exists(f"/proc/{self.pid}") and open(f"/proc/{self.pid}/stat").read().split()[2] != "Z"
        with open(RAW + "/pids_started.txt", "a") as f:
            f.write(f"{time.strftime('%F %T')} {self.name} pid={self.pid} stopped rc={self.p.returncode} alive_after={alive}\n")
        self.fh.close()
        if alive:
            raise RuntimeError(f"pid {self.pid} still alive")


def stt_server(t, corelist, tag):
    argv = [WH + "/build/bin/whisper-server", "-m", "/mnt/raid0/llm/models/whisper-ggml/ggml-large-v3-turbo.bin",
            "--host", "127.0.0.1", "--port", str(STT_PORT), "--inference-path", "/v1/audio/transcriptions",
            "-t", str(t), "-ng"]
    env = {"HIP_VISIBLE_DEVICES": "-1", "LD_LIBRARY_PATH": WH + "/build/bin:/opt/rocm/lib"}
    return Server(f"stt_t{t}_{tag}", argv, env, corelist, STT_PORT, f"{RAW}/stt_server_t{t}_{tag}.log")


def tts_server(t, corelist, tag):
    argv = [QT + "/build/tts-server",
            "--model", "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf",
            "--codec", "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-tokenizer-12hz-Q8_0.gguf",
            "--alias", "qwen3-tts-12hz-0.6b", "--host", "127.0.0.1", "--port", str(TTS_PORT)]
    env = {"HIP_VISIBLE_DEVICES": "-1", "GGML_BACKEND": "CPU", "SHIM_NPROCS": str(2 * t),
           "LD_PRELOAD": TMP + "/nprocs_shim.so", "LD_LIBRARY_PATH": QT + "/build:/opt/rocm/lib"}
    return Server(f"tts_t{t}_{tag}", argv, env, corelist, TTS_PORT, f"{RAW}/tts_server_t{t}_{tag}.log")


# ---------------------------------------------------------------- clients
def stt_request(wavpath):
    data = open(wavpath, "rb").read()
    b = "----" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + data + \
           (f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n--{b}--\r\n").encode()
    c = http.client.HTTPConnection("127.0.0.1", STT_PORT, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    r = c.getresponse(); out = r.read()
    dt = time.perf_counter() - t0
    txt = json.loads(out).get("text", "") if r.status == 200 else out.decode(errors="replace")
    dur = wav_dur(wavpath)
    return {"kind": "stt", "clip": os.path.basename(wavpath), "audio_s": round(dur, 2), "wall_s": round(dt, 3),
            "rtf": round(dt / dur, 4), "status": r.status, "text_head": txt.strip()[:120], "words": len(txt.split())}


def tts_request(text, fmt, seed=42):
    body = json.dumps({"input": text, "response_format": fmt, "seed": seed})
    c = http.client.HTTPConnection("127.0.0.1", TTS_PORT, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/audio/speech", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    t_first, nbytes, need_buf = None, 0, 0.0
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.perf_counter() - t0
        if t_first is None:
            t_first = now
        # playback started at t_first; audio already delivered before this chunk
        # is nbytes/48000 s. If this chunk arrives later than that audio lasts,
        # a player would have underrun unless it pre-buffered the difference.
        need_buf = max(need_buf, (now - t_first) - nbytes / 48000.0)
        nbytes += len(chunk)
    dt = time.perf_counter() - t0
    audio_s = (nbytes - (44 if fmt == "wav" else 0)) / 2 / 24000
    return {"kind": "tts", "fmt": fmt, "text": "short" if text == TEXT_SHORT else "long", "status": r.status,
            "first_packet_s": round(t_first or -1, 3), "wall_s": round(dt, 3), "audio_s": round(audio_s, 2),
            "prebuffer_needed_s": round(need_buf, 3) if fmt == "pcm" else None,
            "rtf": round(dt / audio_s, 4) if audio_s > 0 else None}


def llm_request(results, key, cap_s=900):
    body = json.dumps({"model": "frontdoor", "max_tokens": 300, "temperature": 0.7, "stream": True,
                       "messages": [{"role": "user", "content":
                                     "Explain in detail, in about 250 words, how a refrigerator works, "
                                     "covering the compressor, condenser, expansion valve and evaporator."}]})
    c = http.client.HTTPConnection("127.0.0.1", 8070, timeout=30)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    t_first, n_chunks, usage, buf = None, 0, None, b""
    capped = False
    while True:
        if time.perf_counter() - t0 > cap_s:
            capped = True
            c.close()  # client disconnect -> llama-server cancels the generation
            break
        try:
            chunk = r.read1(65536)
        except (TimeoutError, OSError):
            continue
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            if d.get("timings"):
                results[key + "_timings"] = d["timings"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                n_chunks += 1
                if t_first is None:
                    t_first = time.perf_counter() - t0
    dt = time.perf_counter() - t0
    results[key] = {"status": r.status, "ttft_s": round(t_first or -1, 3), "wall_s": round(dt, 3),
                    "content_chunks": n_chunks, "usage": usage, "capped_at_s": cap_s if capped else None,
                    "chunks_per_s_after_first": round((n_chunks - 1) / (dt - t_first), 2) if t_first and n_chunks > 1 else None}


# ---------------------------------------------------------------- phases
def emit(res, fh, jl):
    jl.write(json.dumps(res) + "\n"); jl.flush()
    log(json.dumps(res), fh)


def parse_spec(spec):
    # "T" -> T threads on cores 0..T-1 ; "T@a-b" -> T threads on core list a-b
    out = []
    for x in spec.split(","):
        t, _, cl = x.partition("@")
        out.append((int(t), cl or cores(0, int(t))))
    return out


def phase_stt_sweep(spec):
    fh = open(RAW + "/stt_sweep.log", "a"); jl = open(RAW + "/stt_sweep.jsonl", "a")
    for t, cl in parse_spec(spec):
        s = stt_server(t, cl, "solo_" + cl)
        try:
            log(f"STT t={t} cores={cl} pid={s.pid} ready in {s.wait_ready():.1f}s", fh)
            stt_request(SHORT_WAV)  # warm-up
            for _ in range(3):
                r = stt_request(SHORT_WAV); r.update(threads=t, cores=cl, mode="solo"); emit(r, fh, jl)
            for _ in range(2):
                r = stt_request(LONG_WAV); r.update(threads=t, cores=cl, mode="solo"); emit(r, fh, jl)
        finally:
            s.stop()


def phase_tts_sweep(spec):
    fh = open(RAW + "/tts_sweep.log", "a"); jl = open(RAW + "/tts_sweep.jsonl", "a")
    for t, cl in parse_spec(spec):
        s = tts_server(t, cl, "solo_" + cl)
        try:
            log(f"TTS t={t} cores={cl} pid={s.pid} ready in {s.wait_ready():.1f}s", fh)
            tts_request(TEXT_SHORT, "pcm")  # warm-up
            for text, n in ((TEXT_SHORT, 3), (TEXT_LONG, 2)):
                for fmt in ("pcm", "wav"):
                    for _ in range(n):
                        r = tts_request(text, fmt); r.update(threads=t, cores=cl, mode="solo"); emit(r, fh, jl)
        finally:
            s.stop()


def phase_concurrent(stt_spec, tts_spec, reps, tag="conc"):
    (stt_t, stt_cl), = parse_spec(stt_spec)
    (tts_t, tts_cl), = parse_spec(tts_spec)
    fh = open(RAW + "/concurrent.log", "a"); jl = open(RAW + "/concurrent.jsonl", "a")
    s = stt_server(stt_t, stt_cl, tag)
    q = tts_server(tts_t, tts_cl, tag)
    try:
        s.wait_ready(); q.wait_ready()
        log(f"CONC[{tag}] stt t={stt_t} cores={stt_cl} pid={s.pid} | tts t={tts_t} cores={tts_cl} pid={q.pid}", fh)
        stt_request(SHORT_WAV); tts_request(TEXT_SHORT, "pcm")  # warm-up
        base = dict(tag=tag, stt_threads=stt_t, stt_cores=stt_cl, tts_threads=tts_t, tts_cores=tts_cl)

        def put(r, mode, rep):
            r.update(base); r.update(mode=mode, rep=rep); emit(r, fh, jl)

        # solo baselines at the SAME layout (servers both resident, only one active)
        for rep in range(reps):
            for clip in (SHORT_WAV, SHORT_WAV, LONG_WAV):
                put(stt_request(clip), "solo-stt", rep)
            for text in (TEXT_SHORT, TEXT_LONG):
                put(tts_request(text, "pcm"), "solo-tts", rep)
            llm = {}; llm_request(llm, "llm"); llm["kind"] = "llm"; put(llm, "solo-llm", rep)

        for mode in ("stt+tts", "stt+tts+llm"):
            for rep in range(reps):
                stop = threading.Event(); out = []; llms = []

                def stt_loop():
                    while not stop.is_set():
                        for clip in (SHORT_WAV, LONG_WAV, SHORT_WAV):
                            if stop.is_set():
                                break
                            out.append(stt_request(clip))

                def llm_loop():
                    # ONE generation in flight at a time, back to back, for the whole window
                    while not stop.is_set():
                        d = {}; llm_request(d, "llm"); d["kind"] = "llm"; llms.append(d)

                def tts_loop():
                    for text in (TEXT_SHORT, TEXT_LONG, TEXT_SHORT):
                        out.append(tts_request(text, "pcm"))

                th = [threading.Thread(target=stt_loop)]
                if mode.endswith("llm"):
                    th.append(threading.Thread(target=llm_loop))
                for t_ in th:
                    t_.start()
                time.sleep(0.5)
                tts_loop()
                stop.set()
                for t_ in th:
                    t_.join()
                for r in out + llms:
                    put(r, mode, rep)
    finally:
        s.stop(); q.stop()


def phase_lean_llm(stt_spec, tts_spec, cap_s, tag):
    """Bounded probe: small speech footprint + ONE frontdoor generation, capped at cap_s."""
    (stt_t, stt_cl), = parse_spec(stt_spec)
    (tts_t, tts_cl), = parse_spec(tts_spec)
    fh = open(RAW + "/concurrent.log", "a"); jl = open(RAW + "/concurrent.jsonl", "a")
    s = stt_server(stt_t, stt_cl, tag); q = tts_server(tts_t, tts_cl, tag)
    try:
        s.wait_ready(); q.wait_ready()
        log(f"LEAN[{tag}] stt t={stt_t} cores={stt_cl} pid={s.pid} | tts t={tts_t} cores={tts_cl} pid={q.pid}", fh)
        stt_request(SHORT_WAV); tts_request(TEXT_SHORT, "pcm")
        base = dict(tag=tag, stt_threads=stt_t, stt_cores=stt_cl, tts_threads=tts_t, tts_cores=tts_cl)

        def put(r, mode, rep):
            r.update(base); r.update(mode=mode, rep=rep); emit(r, fh, jl)
        for _ in range(2):
            put(stt_request(SHORT_WAV), "solo-stt", 0)
            put(tts_request(TEXT_SHORT, "pcm"), "solo-tts", 0)
        d = {}; llm_request(d, "llm", cap_s); d["kind"] = "llm"; put(d, "solo-llm", 0)
        out = []
        th = [threading.Thread(target=lambda: out.append(stt_request(SHORT_WAV))),
              threading.Thread(target=lambda: out.append(tts_request(TEXT_SHORT, "pcm")))]
        d = {}; tl = threading.Thread(target=llm_request, args=(d, "llm", cap_s)); tl.start(); time.sleep(0.5)
        for t_ in th:
            t_.start()
        for t_ in th:
            t_.join()
        tl.join(); d["kind"] = "llm"
        for r in out + [d]:
            put(r, "stt+tts+llm", 0)
    finally:
        s.stop(); q.stop()


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "stt_sweep":
        phase_stt_sweep(sys.argv[2])
    elif ph == "tts_sweep":
        phase_tts_sweep(sys.argv[2])
    elif ph == "lean_llm":
        phase_lean_llm(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
    elif ph == "concurrent":
        phase_concurrent(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "conc")
