"""CS-47 latency leg: per-decision CPU latency (incl. featurisation) with A/A dispersion. python -I lat_worker.py.
Runs INSIDE the exclusive bench claim and a taskset pin (set by latency.sh). Single-threaded graphs (intra_op 1)."""
import argparse, json, sys, time, wave, os, platform
from pathlib import Path
import numpy as np
W = Path("/mnt/raid0/llm/worktrees/cs47-research-20261020")
sys.path.insert(0, str(W / "scripts/voice/turn_taking")); sys.path.insert(0, str(W / "scripts/speech"))
T = Path("/mnt/raid0/llm/tmp/cs-47-20261020"); CS46 = Path("/mnt/raid0/llm/tmp/cs-46-20261020")
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["silero", "smartturn", "parakeet"])
ap.add_argument("--variant", default="fp32_fp32,160,decoder"); ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--clips", type=int, default=6); ap.add_argument("--pin-label", required=True)
ap.add_argument("--aa-pass", required=True); ap.add_argument("--deadline-epoch", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ids = [l.strip() for l in open(T / "all_clips.txt") if l.strip()]
step = max(1, len(ids) // a.clips); ids = ids[::step][:a.clips]
def load(cid):
    with wave.open(str(T / "clips" / f"{cid}.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
    return x[:, 0].astype(np.float32) / 32768.0
clipsx = {c: load(c) for c in ids}
lat_wall, lat_cpu = [], []
def rec(f):
    t0, c0 = time.perf_counter(), time.thread_time(); f(); lat_wall.append(time.perf_counter() - t0); lat_cpu.append(time.thread_time() - c0)
aborted = False
def over(): return time.time() > a.deadline_epoch
if a.arm == "silero":
    import tt_silero
    m = tt_silero.PinnedSilero()
    for r in range(a.repeats):
        for c, x in clipsx.items():
            if over(): aborted = True; break
            s = m.stream()
            for i in range(0, len(x) - 511, 512):
                rec(lambda: s.feed(x[i:i + 512]))
        if aborted: break
elif a.arm == "smartturn":
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    so = ort.SessionOptions(); so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.inter_op_num_threads = 1; so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(T / "dl/smart-turn-v3.2-cpu.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    fe = WhisperFeatureExtractor(chunk_length=8)
    segs = [x[: int(len(x) * f)][-8 * 16000:] for x in clipsx.values() for f in (0.5, 0.75, 1.0)]
    def dec(seg):
        f = fe(seg, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000, truncation=True, do_normalize=True).input_features.astype(np.float32)
        sess.run(None, {"input_features": f})
    for seg in segs[:2]: dec(seg)       # warm-up, not recorded
    for r in range(a.repeats):
        for seg in segs:
            if over(): aborted = True; break
            rec(lambda: dec(seg))
        if aborted: break
else:
    from parakeet_eou.engine import OrtGraphs, ParakeetEOUArm, load_vocab
    from parakeet_eou.parity import PAIRINGS
    pairing, chunk, reset = a.variant.split(","); chunk = int(chunk)
    dec = PAIRINGS[pairing]; DL = CS46 / "dl"; EX = CS46 / "export"
    decp = EX / dec[8:] if dec.startswith("@export/") else DL / dec
    g = OrtGraphs(EX / f"encoder_stream_{chunk}ms.onnx", decp, threads=1); vocab = load_vocab(DL / "vocab.txt")
    n = chunk * 16
    for r in range(a.repeats):
        for c, x in clipsx.items():
            arm = ParakeetEOUArm(g, vocab, chunk_ms=chunk, reset=reset, hold_s=0.0)
            for i in range(0, len(x) - n + 1, n):
                if over(): aborted = True; break
                rec(lambda: arm.feed(x[i:i + n]))     # one decision = one chunk: featurise NEW frames + encoder + greedy decode
            if aborted: break
        if aborted: break
def pct(v, p): return float(np.percentile(v, p)) * 1e3 if v else None
res = dict(arm=a.arm, variant=a.variant if a.arm == "parakeet" else None, pin=a.pin_label, aa_pass=a.aa_pass, n_decisions=len(lat_wall),
           repeats=a.repeats, clips=ids, aborted_by_deadline=aborted,
           wall_ms=dict(p50=pct(lat_wall, 50), p95=pct(lat_wall, 95), p99=pct(lat_wall, 99)), cpu_ms=dict(p50=pct(lat_cpu, 50), p95=pct(lat_cpu, 95)),
           taskset=os.sched_getaffinity(0).__len__(), host=platform.node(), t_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))
