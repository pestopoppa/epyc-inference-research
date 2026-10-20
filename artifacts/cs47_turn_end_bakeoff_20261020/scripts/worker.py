"""CS-47 accuracy-leg worker: one clip -> raw per-arm signals (JSON line). Single thread. python -I."""
import json, sys, time, wave, argparse
from pathlib import Path
import numpy as np
W = Path("/mnt/raid0/llm/worktrees/cs47-research-20261020")
sys.path.insert(0, str(W / "scripts/voice/turn_taking")); sys.path.insert(0, str(W / "scripts/speech"))
T = Path("/mnt/raid0/llm/tmp/cs-47-20261020"); CS46 = Path("/mnt/raid0/llm/tmp/cs-46-20261020")
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--clips", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--variant", default="")
a = ap.parse_args()

def load(cid):
    with wave.open(str(T / "clips" / f"{cid}.wav")) as w:
        n = w.getnframes(); x = np.frombuffer(w.readframes(n), np.int16).reshape(-1, w.getnchannels())
    return x[:, 0].astype(np.float32) / 32768.0

def silero_probs(x):
    import tt_silero
    m = tt_silero.PinnedSilero(); st = m.stream(); P = []
    for i in range(0, len(x) - 511, 512): P.append(float(st.feed(x[i:i + 512])))
    return P

ids = [l.strip() for l in open(a.clips) if l.strip()]
fo = open(a.out, "w")
if a.arm == "silero":
    for cid in ids:
        fo.write(json.dumps({"clip": cid, "arm": "silero", "probs": silero_probs(load(cid))}) + "\n"); fo.flush()
elif a.arm == "smartturn":
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    so = ort.SessionOptions(); so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.inter_op_num_threads = 1; so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(T / "dl/smart-turn-v3.2-cpu.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    fe = WhisperFeatureExtractor(chunk_length=8)
    GATES = [0.2, 0.4, 0.6, 0.8, 1.0]
    for cid in ids:
        x = load(cid); P = silero_probs(x)
        DT = 512 / 16000
        def sfires(stop_s, thr=0.5, start_s=0.1):
            f = []; cons = 0; ins = False; last = 0.0
            for i, pp in enumerate(P):
                t = (i + 1) * DT
                if pp >= thr:
                    cons += 1
                    if cons * DT >= start_s: ins = True
                    if ins: last = t
                else:
                    cons = 0
                    if ins and t - last >= stop_s - 1e-9:
                        f.append((round(t, 4), last)); ins = False
            return f
        rows = []
        for g in GATES:
            for tf, ls in sfires(g):
                n = int(round(tf * 16000))
                if n > len(x): continue
                seg = x[:n][-8 * 16000:]
                t0 = time.perf_counter()
                f = fe(seg, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000, truncation=True, do_normalize=True).input_features.astype(np.float32)
                pr = float(sess.run(None, {"input_features": f})[0].reshape(-1)[0])
                rows.append({"sil_start_s": ls, "gate_s": g, "t_fire_s": n / 16000, "prob": pr, "wall_s": time.perf_counter() - t0})
        fo.write(json.dumps({"clip": cid, "arm": "smartturn", "calls": rows}) + "\n"); fo.flush()
elif a.arm == "parakeet":
    from parakeet_eou.engine import OrtGraphs, ParakeetEOUArm, load_vocab
    from parakeet_eou.parity import PAIRINGS
    pairing, chunk, reset = a.variant.split(",")
    chunk = int(chunk)
    dec = PAIRINGS[pairing]; DL = CS46 / "dl"; EX = CS46 / "export"
    decp = EX / dec[8:] if dec.startswith("@export/") else DL / dec
    g = OrtGraphs(EX / f"encoder_stream_{chunk}ms.onnx", decp)
    vocab = load_vocab(DL / "vocab.txt")
    for cid in ids:
        x = load(cid)
        arm = ParakeetEOUArm(g, vocab, chunk_ms=chunk, reset=reset, hold_s=1.0)
        ev = []; t0 = time.perf_counter()
        for i in range(0, len(x), 1280):
            ev += arm.feed(x[i:i + 1280])
        ev += arm.finish()
        wall = time.perf_counter() - t0
        slim = [{k: e[k] for k in ("t", "kind", "enc_frame") if k in e} | ({"released_t": e["released_t"]} if "released_t" in e else {}) for e in ev if e["kind"] in ("eou", "eob", "reset")]
        ntok = sum(1 for e in ev if e["kind"] == "token")
        fo.write(json.dumps({"clip": cid, "arm": "parakeet", "variant": a.variant, "events": slim, "n_tokens": ntok, "wall_s": wall, "audio_s": len(x) / 16000, "config": arm.config}) + "\n"); fo.flush()
