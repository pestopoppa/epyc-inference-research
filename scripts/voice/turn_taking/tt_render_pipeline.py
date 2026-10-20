#!/usr/bin/env python3
"""CS-45 synthesized-stratum renderer: production qwentts.cpp (READ-ONLY) -> 24 kHz -> 16 kHz, plus
whisper.cpp (production-speech-v1, CPU) word timing for the cut points. CLI processes only; no
server is started and no GPU is touched. Run it under a region-lock BUILD claim on the cores it is
given, e.g.

  region-lock run --regions q3 --role build --tag cs45-render -- \
      python3 tt_render_pipeline.py --scratch /mnt/raid0/llm/tmp/cs45/run1 --cores 72-95 \
          --tts-cpus 73-77,79-81 --tts-threads 8 --asr-layouts '8:73-77,79-81|16:73-77,79-83,85-89,91'

Every child PID is appended to <scratch>/pids_started.txt and every child is reaped (TERM, then KILL,
then verified dead) on timeout or error. Nothing is left running.

Outputs under --scratch:
  raw24/<id>.wav     qwen-tts native 24 kHz mono output
  rec16/<id>.wav     pinned 3:2 polyphase resample (tt_audio.resample_24k_to_16k)
  asr/<id>.json      raw whisper-cli -ojf output (token offsets and, if supported, DTW)
  tts/<id>.log       qwen-tts stderr/stdout
  renders.jsonl      one provenance row per render (final attempt) + attempt history
  linkage_*.txt      verify_ggml_linkage.sh proofs
  env.json           binary/model digests, versions, layout decisions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_plan as P  # noqa: E402

QWEN = "/mnt/raid0/llm/qwentts.cpp/build"
WHISPER = "/mnt/raid0/llm/whisper.cpp/build/bin"
TTS_MODEL = "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf"
TTS_CODEC = "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-tokenizer-12hz-Q8_0.gguf"
ASR_MODEL = "/mnt/raid0/llm/models/whisper-ggml/ggml-large-v3-turbo.bin"
LINKAGE = "/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
SHIM = "/mnt/raid0/llm/tmp/cs45/nprocs_shim.so"
MAX_ATTEMPTS = 3


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


class Runner:
    def __init__(self, pidfile: Path):
        self.pidfile = pidfile

    def run(self, argv, env, stdin_text=None, out_path=None, timeout=180):
        """Run one child; returns (rc, seconds). Always reaps; never leaves a process behind."""
        out_f = open(out_path, "wb") if out_path else subprocess.DEVNULL
        t0 = time.time()
        p = subprocess.Popen(
            argv, env=env, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=out_f, stderr=subprocess.STDOUT, start_new_session=True,
        )
        with open(self.pidfile, "a") as pf:
            pf.write(f"{time.strftime('%FT%TZ', time.gmtime())} pid={p.pid} {' '.join(argv[:3])} ...\n")
        try:
            if stdin_text is not None:
                p.communicate(stdin_text.encode(), timeout=timeout)
            else:
                p.wait(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -999
            self._kill(p)
        except BaseException:
            self._kill(p)
            raise
        finally:
            if out_path:
                out_f.close()
        return rc, time.time() - t0

    @staticmethod
    def _kill(p):
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                p.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        try:
            os.kill(p.pid, 0)
            raise RuntimeError(f"child {p.pid} still alive after SIGKILL")
        except ProcessLookupError:
            pass


def tts_env(threads: int) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    e.update({"HIP_VISIBLE_DEVICES": "-1", "GGML_BACKEND": "CPU", "LD_LIBRARY_PATH": f"{QWEN}:/opt/rocm/lib",
              "SHIM_NPROCS": str(2 * threads), "LD_PRELOAD": SHIM})
    return e


def asr_env() -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    e.update({"HIP_VISIBLE_DEVICES": "-1", "LD_LIBRARY_PATH": f"{WHISPER}:/opt/rocm/lib"})
    return e


def parse_asr(path: Path) -> dict:
    """whisper-cli -ojf -> {'text': str, 'words': [{w, t0_ms, t1_ms, dtw_ms|None, p}]}"""
    d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    words, cur = [], None
    text = []
    for seg in d.get("transcription", []):
        text.append(seg.get("text", ""))
        for tk in seg.get("tokens", []):
            t = tk.get("text", "")
            if t.startswith("[_") or t.startswith("<|") or not t.strip():
                continue
            off = tk.get("offsets", {})
            dtw = tk.get("t_dtw", -1)
            dtw_ms = int(dtw) * 10 if isinstance(dtw, (int, float)) and dtw >= 0 else None
            if cur is None or t.startswith(" "):
                cur = {"w": t.strip(), "t0_ms": off.get("from"), "t1_ms": off.get("to"), "dtw_ms": dtw_ms, "p": tk.get("p")}
                words.append(cur)
            else:
                cur["w"] += t
                cur["t1_ms"] = off.get("to")
                if cur["dtw_ms"] is None:
                    cur["dtw_ms"] = dtw_ms
    return {"text": "".join(text).strip(), "words": words}


def wer(ref: list[str], hyp: list[str]) -> float:
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[m] / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--cores", required=True, help="claimed region, for the record only, e.g. 72-95")
    ap.add_argument("--tts-cpus", required=True, help="taskset cpu list for qwen-tts, chosen so no SMT sibling is busy")
    ap.add_argument("--tts-threads", type=int, default=8)
    ap.add_argument("--asr-layouts", required=True, help="'threads:cpulist|threads:cpulist' tried in order (probe must finish in 25 s)")
    ap.add_argument("--only", default="", help="comma list of render_ids (re-render subset)")
    ap.add_argument("--seed-bump", type=int, default=0, help="added to attempt index for the seeds of --only renders")
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()

    S = Path(args.scratch)
    for d in ("raw24", "rec16", "asr", "tts"):
        (S / d).mkdir(parents=True, exist_ok=True)
    runner = Runner(S / "pids_started.txt")
    plan = P.build_plan()
    jobs = plan["jobs"]
    if args.only:
        want = set(args.only.split(","))
        jobs = [j for j in jobs if j["render_id"] in want]

    # ---- environment digests + linkage proof (read-only use of the frozen speech trees)
    env_json = S / "env.json"
    if not env_json.exists():
        info = {"date_utc": time.strftime("%FT%TZ", time.gmtime()), "claimed_region_cores": args.cores, "tts_cpus": args.tts_cpus,
                "tts_threads": args.tts_threads, "asr_layouts": args.asr_layouts}
        for name, path in (("qwen-tts", f"{QWEN}/qwen-tts"), ("whisper-cli", f"{WHISPER}/whisper-cli")):
            info[name] = {"path": path, "sha256": sha256_file(path), "mtime": time.strftime("%FT%TZ", time.gmtime(os.path.getmtime(path)))}
        for name, path in (("tts_talker", TTS_MODEL), ("tts_codec", TTS_CODEC), ("asr_model", ASR_MODEL), ("nprocs_shim", SHIM)):
            info[name] = {"path": path, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
        env_json.write_text(json.dumps(info, indent=1))
        for name, binp, tree, e in (("qwen-tts", f"{QWEN}/qwen-tts", QWEN, tts_env(args.tts_threads)),
                                    ("whisper-cli", f"{WHISPER}/whisper-cli", WHISPER, asr_env())):
            e = dict(e)
            e.pop("LD_PRELOAD", None)
            rc, _ = runner.run([LINKAGE, binp, tree], e, out_path=str(S / f"linkage_{name}.txt"), timeout=120)
            if rc != 0:
                print(f"LINKAGE FAIL {name} rc={rc}; see linkage_{name}.txt", file=sys.stderr)
                return 3
        rc, _ = runner.run([f"{QWEN}/qwen-tts", "--help"], tts_env(args.tts_threads), out_path=str(S / "qwen-tts.help.txt"), timeout=30)
        rc, _ = runner.run([f"{WHISPER}/whisper-cli", "--help"], asr_env(), out_path=str(S / "whisper-cli.help.txt"), timeout=30)

    tts_cores = args.tts_cpus
    # whisper layout: a probe that does not finish in 25 s marks a layout as collapsed
    # (speech_cpu_realtime_20260924 hazard 1) and the next one is tried.
    layouts = [(int(x.split(":")[0]), x.split(":")[1]) for x in args.asr_layouts.split("|")]
    asr_layout = None
    dtw_ok = True
    rows_path = S / "renders.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done[r["render_id"]] = r

    def asr_cmd(wav: Path, base: Path, threads: int, cores: str, dtw: bool):
        argv = ["taskset", "-c", cores, f"{WHISPER}/whisper-cli", "-m", ASR_MODEL, "-f", str(wav), "-t", str(threads),
                "-ng", "-l", "en", "-ojf", "-of", str(base), "-np"]
        if dtw:
            argv += ["--dtw", "large.v3.turbo"]
        return argv

    def do_asr(wav: Path, rid: str):
        nonlocal asr_layout, dtw_ok
        base = S / "asr" / rid
        if asr_layout is None:
            for th, cs in layouts:
                for dtw in (True, False):
                    rc, dt = runner.run(asr_cmd(wav, base, th, cs, dtw), asr_env(), out_path=str(S / "asr" / f"{rid}.log"), timeout=25)
                    if rc == 0 and (S / "asr" / f"{rid}.json").exists() and dt < 20:
                        asr_layout = (th, cs)
                        dtw_ok = dtw
                        (S / "asr_layout.json").write_text(json.dumps({"threads": th, "cores": cs, "dtw": dtw, "probe_s": round(dt, 2)}))
                        return parse_asr(S / "asr" / f"{rid}.json")
                    (S / "asr" / f"{rid}.json").unlink(missing_ok=True)
            raise RuntimeError("no healthy whisper layout found in the claimed cores")
        th, cs = asr_layout
        rc, dt = runner.run(asr_cmd(wav, base, th, cs, dtw_ok), asr_env(), out_path=str(S / "asr" / f"{rid}.log"), timeout=120)
        if rc != 0 or not (S / "asr" / f"{rid}.json").exists():
            raise RuntimeError(f"asr failed for {rid} rc={rc}")
        return parse_asr(S / "asr" / f"{rid}.json")

    # ---- phased execution: (A) parallel single-thread TTS, (B) sequential ASR, (C) QA and retry.
    # qwen-tts is decode-latency-bound and its per-graph thread spawn collapses under host load (measured on
    # this host 2026-10-20: 8 threads = 472 ms/frame, 1 thread = 36 ms/frame), so jobs run as independent
    # single-thread processes, one per core.
    from concurrent.futures import ThreadPoolExecutor

    cpu_list = []
    for part in args.tts_cpus.split(","):
        lo_, _, hi_ = part.partition("-")
        cpu_list += list(range(int(lo_), int(hi_ or lo_) + 1))
    state = {j["render_id"]: {"job": j, "history": [], "ok": False, "asr": None} for j in jobs}
    for rid, r in done.items():
        if rid in state and r.get("qa_ok") and not args.only:
            state[rid].update(ok=True, history=r.get("history", []), asr=r.get("asr"), row=r)

    def tts_one(rid: str, cpu: int, attempt: int):
        j = state[rid]["job"]
        seed = (j["seed"] + 7919 * (attempt + args.seed_bump)) % 2_000_000_000
        raw = S / "raw24" / f"{rid}.wav"
        raw.unlink(missing_ok=True)
        argv = ["taskset", "-c", str(cpu), f"{QWEN}/qwen-tts", "--model", TTS_MODEL, "--codec", TTS_CODEC, "--lang", "English",
                "--seed", str(seed), "--ref-wav", j["ref"], "-o", str(raw)]
        rc, dt = runner.run(argv, tts_env(1), stdin_text=j["text"] + "\n", out_path=str(S / "tts" / f"{rid}.log"), timeout=600)
        if rc != 0 or not raw.exists():
            return {"attempt": attempt, "seed": seed, "tts_rc": rc, "qa": "tts-failed"}
        p24 = A.read_wav(raw)
        if p24.sample_rate != 24000 or p24.channels != 1:
            return {"attempt": attempt, "seed": seed, "qa": f"unexpected format {p24.sample_rate}/{p24.channels}"}
        x16 = A.resample_24k_to_16k(p24.samples[:, 0])
        A.write_wav(S / "rec16" / f"{rid}.wav", A.Pcm(x16[:, None], A.SR))
        return {"attempt": attempt, "seed": seed, "tts_s": round(dt, 2), "qa": "asr-pending"}

    def qa(rid: str, asr: dict, hist: dict):
        j = state[rid]["job"]
        ref = P.norm_words(j["text"])
        hyp = P.norm_words(asr["text"])
        w = wer(ref, hyp)
        dur = A.read_wav(S / "rec16" / f"{rid}.wav").n_samples / A.SR
        if j.get("family") == "CTX_BC":  # acknowledgement phrases: duration/non-empty only
            good = 0.25 <= dur <= 2.0 and len(hyp) > 0
        else:
            good = w <= 0.15 and 0.2 <= dur / max(1, len(ref)) <= 0.9
        hist["qa"] = f"{'ok' if good else 'FAIL'} wer={w:.3f} dur={dur:.2f}"
        return good

    pending = [rid for rid, st in state.items() if not st["ok"]]
    for attempt in range(MAX_ATTEMPTS):
        if not pending:
            break
        print(f"attempt {attempt}: TTS for {len(pending)} renders on {len(cpu_list)} single-thread workers", flush=True)
        # one worker per core: round-robin assignment, each worker runs its share sequentially
        shares = {c: [] for c in cpu_list}
        for i, rid in enumerate(pending):
            shares[cpu_list[i % len(cpu_list)]].append(rid)

        def worker(cpu):
            return [(rid, tts_one(rid, cpu, attempt)) for rid in shares[cpu]]

        with ThreadPoolExecutor(max_workers=len(cpu_list)) as ex:
            for res in ex.map(worker, cpu_list):
                for rid, h in res:
                    state[rid]["history"].append(h)
        if attempt == 0 and all(state[r]["history"][-1]["qa"] == "tts-failed" for r in pending):
            print("every first-attempt TTS job failed; aborting (see tts/*.log)", file=sys.stderr)
            return 4
        nxt = []
        for rid in pending:
            h = state[rid]["history"][-1]
            if h["qa"] == "tts-failed" or h["qa"].startswith("unexpected"):
                nxt.append(rid)
                continue
            if args.no_asr:
                state[rid]["ok"] = True
                continue
            asr = do_asr(S / "rec16" / f"{rid}.wav", rid)
            state[rid]["asr"] = asr
            if qa(rid, asr, h):
                state[rid]["ok"] = True
            else:
                nxt.append(rid)
            print(f"{rid} {h['qa']}", flush=True)
        pending = nxt
    out = []
    for rid, st in state.items():
        if "row" in st and st["ok"]:
            out.append(st["row"])
            continue
        j = st["job"]
        row = {"render_id": rid, "role": j["role"], "text": j["text"], "voice_id": j["voice_id"], "ref_wav": j["ref"],
               "ref_sha256": sha256_file(j["ref"]), "split": j["split"], "family": j.get("family"), "pair_id": j.get("pair_id"),
               "prefix_words": j.get("prefix_words"), "seed_final": st["history"][-1].get("seed") if st["history"] else None,
               "history": st["history"], "qa_ok": st["ok"]}
        if (S / "raw24" / f"{rid}.wav").exists() and (S / "rec16" / f"{rid}.wav").exists():
            row["raw24_pcm_sha256"] = A.read_wav(S / "raw24" / f"{rid}.wav").pcm_sha256()
            r16 = A.read_wav(S / "rec16" / f"{rid}.wav")
            row["rec16_pcm_sha256"] = r16.pcm_sha256()
            row["rec16_samples"] = r16.n_samples
        if st["asr"]:
            row["asr"] = st["asr"]
        out.append(row)
        done[rid] = row
    with open(rows_path, "w") as f:
        for r in done.values():
            f.write(json.dumps(r) + "\n")
    print(f"done: {sum(1 for r in out if r['qa_ok'])}/{len(out)} renders passed QA", flush=True)
    return 0 if all(r["qa_ok"] for r in out) else 5


if __name__ == "__main__":
    sys.exit(main())
