#!/usr/bin/env python3
"""Addendum 3: LIVE production speech (:9000 whisper 0-23, :9002 qwentts 24-39) under CPU LLM load.

Does NOT start or stop any speech service. Sends:
  * STT: POST :9000/v1/audio/transcriptions (jfk 11 s, 86.5 s concat clip)
  * TTS: POST :9002/v1/audio/speech (pcm stream: first packet + RTF)
  * LLM: ONE generation in flight at a time (ignore_eos, fixed max_tokens) to a chosen port,
         back to back for the duration of the speech window.
Speech requests are issued one at a time (an "urgent request"), so each one is measured alone
against the LLM, not against the other speech service.

usage: live_llm_contention.py <tag> <llm_port|none> <max_tokens> <reps>
"""
import http.client, json, os, socket, sys, threading, time, uuid
import speech_cpu_bench as B

B.STT_PORT, B.TTS_PORT = 9000, 9002
TAG = sys.argv[1]
LLM_PORT = None if sys.argv[2] == "none" else int(sys.argv[2])
MAXTOK = int(sys.argv[3])
REPS = int(sys.argv[4])
T0 = time.time()
LLM_CAP = float(os.environ.get("LLM_CAP_S", "180"))
REQ_CAP = float(os.environ.get("REQ_CAP_S", "90"))
fh = open(B.RAW + "/live_contention.log", "a")
jl = open(B.RAW + "/live_contention.jsonl", "a")
PROMPT = ("Write a long, detailed technical essay on the history of numerical weather prediction, "
          "from Richardson's forecast factory to modern ensemble systems. Use many sections.")


def llm_once(port, maxtok):
    body = json.dumps({"max_tokens": maxtok, "temperature": 0.7, "stream": True, "ignore_eos": True,
                       "messages": [{"role": "user", "content": PROMPT}]})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    ts = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    buf, timings, t_first, capped, n_tok, t_last = b"", None, None, False, 0, None
    while True:
        if time.time() - ts > LLM_CAP:  # hard cap: never hold a production LLM long
            capped = True; c.close(); break
        try:
            ch = r.read1(65536)
        except OSError:
            continue
        if not ch:
            break
        if t_first is None:
            t_first = time.time() - ts
        buf += ch
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except ValueError:
                continue
            if d.get("timings"):
                timings = d["timings"]
            chs = d.get("choices") or []
            if chs and (chs[0].get("delta") or {}).get("content"):
                n_tok += 1; t_last = time.time()
    te = time.time()
    return {"kind": "llm", "port": port, "t_start": round(ts - T0, 2), "t_end": round(te - T0, 2),
            "wall_s": round(te - ts, 2), "ttft_s": round(t_first or -1, 3), "capped": capped,
            "predicted_n": (timings or {}).get("predicted_n"),
            "tok_s": round((timings or {}).get("predicted_per_second", 0), 2) or None,
            "stream_chunks": n_tok,
            "stream_chunks_per_s": round((n_tok - 1) / (t_last - ts - t_first), 2) if n_tok > 1 and t_last else None,
            "draft_acc": f"{(timings or {}).get('draft_n_accepted')}/{(timings or {}).get('draft_n')}"}


def emit(r):
    r.update(tag=TAG, llm_port=LLM_PORT)
    jl.write(json.dumps(r) + "\n"); jl.flush()
    B.log(json.dumps(r), fh)


def stt_capped(wav):
    try:
        c0 = http.client.HTTPConnection
        class C(c0):
            def __init__(self, h, p, timeout=None):
                super().__init__(h, p, timeout=REQ_CAP)
        B.http.client.HTTPConnection = C
        try:
            return B.stt_request(wav)
        finally:
            B.http.client.HTTPConnection = c0
    except (socket.timeout, TimeoutError):
        d = B.wav_dur(wav)
        return {"kind": "stt", "clip": os.path.basename(wav), "audio_s": round(d, 2), "wall_s": None,
                "rtf": None, "timed_out_after_s": REQ_CAP, "rtf_lower_bound": round(REQ_CAP / d, 2)}


def tts_capped(text):
    body = json.dumps({"input": text, "response_format": "pcm", "seed": 42})
    c = http.client.HTTPConnection("127.0.0.1", B.TTS_PORT, timeout=REQ_CAP)
    t0 = time.perf_counter()
    c.request("POST", "/v1/audio/speech", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    t_first, n, timed_out = None, 0, False
    while True:
        if time.perf_counter() - t0 > REQ_CAP:
            timed_out = True; c.close(); break  # disconnect -> sink returns false -> synthesis aborts
        try:
            ch = r.read1(65536)
        except (socket.timeout, TimeoutError):
            timed_out = True; c.close(); break
        if not ch:
            break
        if t_first is None:
            t_first = time.perf_counter() - t0
        n += len(ch)
    dt = time.perf_counter() - t0
    a = n / 48000.0
    return {"kind": "tts", "fmt": "pcm", "text": "short" if text == B.TEXT_SHORT else "long",
            "first_packet_s": round(t_first, 3) if t_first else None, "wall_s": round(dt, 2),
            "audio_s": round(a, 2), "rtf": round(dt / a, 3) if a > 0 and not timed_out else None,
            "timed_out_after_s": REQ_CAP if timed_out else None,
            "audio_delivered_s": round(a, 2)}


def speech(item):
    ts = time.time()
    r = stt_capped(item[1]) if item[0] == "stt" else tts_capped(item[1])
    r["t_start"], r["t_end"] = round(ts - T0, 2), round(time.time() - T0, 2)
    return r


FULL = [("tts", B.TEXT_SHORT), ("stt", B.SHORT_WAV), ("tts", B.TEXT_LONG), ("stt", B.LONG_WAV)]
SHORT = [("tts", B.TEXT_SHORT), ("stt", B.SHORT_WAV)]

B.log(f"LIVE[{TAG}] llm_port={LLM_PORT} max_tokens={MAXTOK} reps={REPS}", fh)
def proc_cpu(pid):
    a = open(f"/proc/{pid}/stat").read().split(); return int(a[13]) + int(a[14])


def wait_speech_idle():
    # read-only /proc sampling of the live speech servers; wait until both are idle
    pids = [int(os.popen(f"ss -ltnp | grep ':{p} ' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2").read() or 0)
            for p in (9000, 9002)]
    t0 = time.time()
    while time.time() - t0 < 900:
        a = [proc_cpu(p) for p in pids]; time.sleep(1.0); b = [proc_cpu(p) for p in pids]
        if all(y - x < 30 for x, y in zip(a, b)):  # < 0.3 cores busy
            return round(time.time() - t0, 1)
    return None


collapsed = 0
for rep in range(REPS):
    B.log(f"speech idle after {wait_speech_idle()} s", fh)
    items = FULL if collapsed == 0 else SHORT
    stop = threading.Event(); llms = []

    def loop():
        while not stop.is_set():
            llms.append(llm_once(LLM_PORT, MAXTOK))

    th = None
    if LLM_PORT:
        th = threading.Thread(target=loop); th.start(); time.sleep(3.0)  # let decode reach steady state
    out = []
    for it in items:
        out.append(speech(it))
        if out[-1].get("timed_out_after_s"):
            break
    stop.set()
    if th:
        th.join()
    for r in out:
        # fraction of this speech request that overlapped an in-flight LLM generation
        ov = sum(max(0, min(r["t_end"], l["t_end"]) - max(r["t_start"], l["t_start"])) for l in llms)
        r["llm_overlap"] = round(ov / max(r["t_end"] - r["t_start"], 1e-9), 3) if LLM_PORT else None
        r["rep"] = rep; emit(r)
    for l in llms:
        l["rep"] = rep; emit(l)
    bad = any(r.get("timed_out_after_s") or (r.get("rtf") or 0) > 1.0 for r in out)
    llm_bad = any((l["tok_s"] or l["stream_chunks_per_s"] or 99) < 5 for l in llms)
    if bad or llm_bad:
        collapsed += 1
    B.log(f"guard rep={rep} speech_realtime_broken={bad} llm_collapsed={llm_bad} collapsed_reps={collapsed}", fh)
    if collapsed > 1:
        B.log("guard: degradation in more than one repetition -> stop", fh)
        break
