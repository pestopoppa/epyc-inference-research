#!/usr/bin/env python3
"""CS-45 synthesized-stratum renderer, v2 (2026-10-20).

Phases (run it under a region-lock BUILD claim on the cores it is given):
  A  TTS: the production qwentts.cpp CLI (READ-ONLY, CPU), ONE SINGLE-THREAD PROCESS PER CORE, in parallel.
     (qwen-tts under host load: 8 threads = 472 ms/frame, 1 thread = 36 ms/frame; see render_logs.)
  B  ALIGN: one faster-whisper large-v3-turbo process (tt_align_fw.py, ASR-A) produces word timestamps + a transcript for
     every new recording. whisper.cpp's token timestamps are NOT used for cut placement any more (root cause of the
     v1 misaligned cuts).
  C  text QA: the ASR-A transcript must equal the intended text (numerals normalised) for cut-family recordings; a
     failing render is re-rendered with the next seed (up to MAX_ATTEMPTS). This is only a pre-filter: the
     independent gates (tt_qa_gates.py, ASR-B) decide admission.

Jobs are the USER recordings of the main plan and of the reserve plan (assistant contexts already exist in the corpus).
A variant k > 0 re-renders the same text with a different seed under id '<render_id>.v<k>' (the builder keeps the first
variant that passes all gates). Every child PID is appended to <scratch>/pids_started.txt and reaped on timeout/error.

  region-lock run --regions q0 --role build --tag cs45-render -- \
     /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -I tt_render_pipeline.py --scratch DIR --variant 0 \
        --tts-cpus 1-23 --align-cpus 0-15 [--only id,id,...]
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_plan as P  # noqa: E402
import tt_qa_gates as Q  # noqa: E402  (norm_words only)

QWEN = "/mnt/raid0/llm/qwentts.cpp/build"
TTS_MODEL = "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf"
TTS_CODEC = "/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-tokenizer-12hz-Q8_0.gguf"
LINKAGE = "/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
SHIM = "/mnt/raid0/llm/tmp/cs45/nprocs_shim.so"
PACE_PY = "/mnt/raid0/llm/pace-env/bin/python"
MAX_ATTEMPTS = 2
HERE = Path(__file__).resolve().parent


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
        out_f = open(out_path, "wb") if out_path else subprocess.DEVNULL
        t0 = time.time()
        p = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL, stdout=out_f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        with open(self.pidfile, "a") as pf:
            pf.write(f"{time.strftime('%FT%TZ', time.gmtime())} pid={p.pid} {' '.join(argv[:4])} ...\n")
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


def tts_env() -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    e.update({"HIP_VISIBLE_DEVICES": "-1", "GGML_BACKEND": "CPU", "LD_LIBRARY_PATH": f"{QWEN}:/opt/rocm/lib", "SHIM_NPROCS": "2", "LD_PRELOAD": SHIM})
    return e


def expand_cpus(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


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


def user_jobs() -> list[dict]:
    jobs = [j for j in P.build_plan()["jobs"] if j["role"] == "user"]
    jobs += [j for j in P.build_reserve_plan()["jobs"] if j["role"] == "user"]
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of base render ids (without .vK)")
    ap.add_argument("--tts-cpus", required=True)
    ap.add_argument("--align-cpus", required=True)
    ap.add_argument("--align-procs", type=int, default=4)
    ap.add_argument("--realign", action="store_true", help="skip TTS for ids whose recording exists and re-run ONLY the ASR-A alignment + cut search")
    a = ap.parse_args()

    S = Path(a.scratch)
    for d in ("raw24", "rec16", "tts", "align"):
        (S / d).mkdir(parents=True, exist_ok=True)
    runner = Runner(S / "pids_started.txt")
    jobs = user_jobs()
    if a.only:
        want = set(a.only.split(","))
        jobs = [j for j in jobs if j["render_id"] in want]
    suffix = f".v{a.variant}" if a.variant else ""
    for j in jobs:
        j["base_id"] = j["render_id"]
        j["render_id"] = j["base_id"] + suffix
        j["variant"] = a.variant

    env_json = S / "env.json"
    if not env_json.exists():
        info = {"date_utc": time.strftime("%FT%TZ", time.gmtime()), "tts_cpus": a.tts_cpus, "align_cpus": a.align_cpus, "align_procs": a.align_procs,
                "qwen-tts": {"path": f"{QWEN}/qwen-tts", "sha256": sha256_file(f"{QWEN}/qwen-tts")}}
        for name, path in (("tts_talker", TTS_MODEL), ("tts_codec", TTS_CODEC), ("nprocs_shim", SHIM)):
            info[name] = {"path": path, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
        env_json.write_text(json.dumps(info, indent=1))
        e = {k: v for k, v in tts_env().items() if k != "LD_PRELOAD"}
        rc, _ = runner.run([LINKAGE, f"{QWEN}/qwen-tts", QWEN], e, out_path=str(S / "linkage_qwen-tts.txt"), timeout=120)
        if rc != 0:
            print("LINKAGE FAIL qwen-tts", file=sys.stderr)
            return 3

    rows_path = S / "renders.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done[r["render_id"]] = r
    state = {j["render_id"]: {"job": j, "history": [], "ok": False, "asr": None} for j in jobs}
    for rid in list(state):
        if rid in done and done[rid].get("qa_ok") and not a.realign:
            state[rid].update(ok=True, history=done[rid]["history"], asr=done[rid].get("asr"), row=done[rid])
    cpus = expand_cpus(a.tts_cpus)

    def tts_one(rid: str, cpu: int, attempt: int):
        j = state[rid]["job"]
        if a.realign and (S / "rec16" / f"{rid}.wav").exists():
            return {"attempt": attempt, "seed": None, "qa": "align-pending", "realign": True}
        seed = (j["seed"] + 7919 * (attempt + 101 * a.variant)) % 2_000_000_000
        raw = S / "raw24" / f"{rid}.wav"
        raw.unlink(missing_ok=True)
        argv = ["taskset", "-c", str(cpu), f"{QWEN}/qwen-tts", "--model", TTS_MODEL, "--codec", TTS_CODEC, "--lang", "English", "--seed", str(seed),
                "--ref-wav", j["ref"], "-o", str(raw)]
        rc, dt = runner.run(argv, tts_env(), stdin_text=j["text"] + "\n", out_path=str(S / "tts" / f"{rid}.log"), timeout=600)
        if rc != 0 or not raw.exists():
            return {"attempt": attempt, "seed": seed, "tts_rc": rc, "qa": "tts-failed"}
        p24 = A.read_wav(raw)
        if p24.sample_rate != 24000 or p24.channels != 1:
            return {"attempt": attempt, "seed": seed, "qa": f"unexpected format {p24.sample_rate}/{p24.channels}"}
        x16 = A.resample_24k_to_16k(p24.samples[:, 0])
        A.write_wav(S / "rec16" / f"{rid}.wav", A.Pcm(x16[:, None], A.SR))
        return {"attempt": attempt, "seed": seed, "tts_s": round(dt, 2), "qa": "align-pending"}

    def align(rids: list[str]) -> dict:
        """ASR-A over all new recordings, sharded across --align-procs processes on disjoint cores."""
        cl = expand_cpus(a.align_cpus)
        nproc = max(1, min(a.align_procs, len(rids), len(cl)))
        per = len(cl) // nproc
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
        env["HIP_VISIBLE_DEVICES"] = "-1"
        jobs_by = {}
        for r in rids:
            j = state[r]["job"]
            row = {"id": r, "wav": str(S / "rec16" / f"{r}.wav"), "text": j["text"]}
            if j.get("family") in ("CUT_IC", "CUT_WAIT"):
                row["n_prefix"] = j["prefix_words"]
            jobs_by[r] = row
        outs = []

        def shard(i):
            sub = rids[i::nproc]
            lst, outp = S / "align" / f"jobs{i}.jsonl", S / "align" / f"out{i}.jsonl"
            lst.write_text("\n".join(json.dumps(jobs_by[r]) for r in sub) + "\n")
            cpuset = ",".join(str(c) for c in cl[i * per : (i + 1) * per])
            rc, dt = runner.run(["taskset", "-c", cpuset, PACE_PY, "-I", str(HERE / "tt_align_fw.py"), "--list", str(lst), "--out", str(outp), "--threads", str(per)],
                                env, out_path=str(S / "align" / f"run{i}.log"), timeout=7200)
            if rc != 0:
                raise RuntimeError(f"alignment shard {i} failed rc={rc}; see {S/'align'/f'run{i}.log'}")
            return outp

        with ThreadPoolExecutor(max_workers=nproc) as ex:
            outs = list(ex.map(shard, range(nproc)))
        res = {}
        for outp in outs:
            for l in outp.read_text().splitlines():
                d = json.loads(l)
                res[d["id"]] = d
        return res

    pending = [rid for rid, st in state.items() if not st["ok"]]
    for attempt in range(MAX_ATTEMPTS):
        if not pending:
            break
        print(f"attempt {attempt}: TTS for {len(pending)} renders on {len(cpus)} single-thread workers", flush=True)
        shares = {c: [] for c in cpus}
        for i, rid in enumerate(pending):
            shares[cpus[i % len(cpus)]].append(rid)

        def worker(cpu):
            return [(rid, tts_one(rid, cpu, attempt)) for rid in shares[cpu]]

        with ThreadPoolExecutor(max_workers=len(cpus)) as ex:
            for res in ex.map(worker, cpus):
                for rid, h in res:
                    state[rid]["history"].append(h)
        if attempt == 0 and all(state[r]["history"][-1]["qa"] == "tts-failed" for r in pending):
            print("every first-attempt TTS job failed; aborting (see tts/*.log)", file=sys.stderr)
            return 4
        ok_tts = [r for r in pending if state[r]["history"][-1]["qa"] == "align-pending"]
        print(f"attempt {attempt}: aligning {len(ok_tts)} recordings with ASR-A (faster-whisper large-v3-turbo)", flush=True)
        res = align(ok_tts) if ok_tts else {}
        nxt = [r for r in pending if r not in ok_tts]
        for rid in ok_tts:
            asr = res[rid]
            j = state[rid]["job"]
            ref = Q.norm_words(j["text"])
            hyp = Q.norm_words(asr["text"])
            w = wer(ref, hyp)
            dur = A.read_wav(S / "rec16" / f"{rid}.wav").n_samples / A.SR
            if j.get("family") == "CTX_BC":
                good = 0.25 <= dur <= 2.0 and len(hyp) > 0
            else:
                good = hyp == ref
            state[rid]["history"][-1]["qa"] = f"{'ok' if good else 'FAIL'} asrA_wer={w:.3f} dur={dur:.2f}"
            state[rid]["asr"] = {"text": asr["text"], "words": asr["words"], "engine": "faster-whisper large-v3-turbo int8 CPU", "cut_candidates": asr.get("cut_candidates")}
            if good:
                state[rid]["ok"] = True
            else:
                nxt.append(rid)
        pending = nxt
    out = []
    for rid, st in state.items():
        if "row" in st and st["ok"]:
            out.append(st["row"])
            continue
        j = st["job"]
        row = {"render_id": rid, "base_id": j["base_id"], "variant": j["variant"], "role": j["role"], "text": j["text"], "voice_id": j["voice_id"],
               "ref_wav": j["ref"], "ref_sha256": sha256_file(j["ref"]), "split": j["split"], "family": j.get("family"), "pair_id": j.get("pair_id"),
               "prefix_words": j.get("prefix_words"), "seed_final": st["history"][-1].get("seed") if st["history"] else None, "history": st["history"],
               "qa_ok": st["ok"]}
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
    print(f"done: {sum(1 for r in out if r['qa_ok'])}/{len(out)} renders passed the ASR-A text pre-filter", flush=True)
    return 0 if all(r["qa_ok"] for r in out) else 5


if __name__ == "__main__":
    sys.exit(main())
