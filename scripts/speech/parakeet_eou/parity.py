"""CS-46 (e): parity gate of the onnxruntime arm against a NeMo CPU streaming run.

Inputs : clip list JSONL ({"clip_id","path",...}) and the ``nemo_ref.py`` output tree
         ``<nemo-root>/c<chunk>_<reset>/`` (per clip ``<id>.json``, ``<id>.mel.npy`` [, ``<id>.va_feats.npy``]).
Outputs: ``parity_report.json`` (per clip: raw token streams of NeMo and of every arm config, EOU step
         indices, feature diffs), ``parity_rows.jsonl`` (one row per clip x config) and a summary.

An arm config is (chunk_ms, reset in {full,none}, frontend in {nemo_va,exact}, graph pairing).
Parity = identical per-step token-id lists (so token stream AND the step/frame at which each token,
including <EOU>, appears).  usage (arm venv):  python -I parity.py --clips C --nemo-root R --model-dir M --export-dir E --out O
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parakeet_eou.engine import OrtGraphs, ParakeetEOUArm, load_vocab  # noqa: E402
from parakeet_eou.frontend import offline_mel  # noqa: E402

PAIRINGS = {   # name: (encoder (per chunk), decoder_joint file in model-dir or export-dir)
    "fp32_fp32": "decoder_joint-model.onnx",
    "fp32_int8dec": "decoder_joint-model.int8.onnx",
    "fp32_selfdec": "@export/decoder_joint_self_fp32.onnx",
    "fp32_selfdec_int8": "@export/decoder_joint_self_int8.onnx",
}


def read_pcm16(path: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == 16000, sr
    return np.clip(np.round(x[:, 0] * 32768.0), -32768, 32767).astype("<i2")


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def run_arm(arm: ParakeetEOUArm, pcm: np.ndarray, block: int) -> dict:
    arm.reset_all()
    events = []
    t0 = time.perf_counter()
    for i in range(0, len(pcm), block):
        events += arm.feed(pcm[i:i + block])
    events += arm.finish()
    return {"steps": arm.log, "events": events, "wall_s": time.perf_counter() - t0,
            "feat_log": np.stack(arm.feat_log) if arm.feat_log else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--nemo-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nemo-suffix", default="", help="'' (shipped, dithered frontend) or '_eval' (deterministic)")
    ap.add_argument("--chunks", default="160,80")
    ap.add_argument("--resets", default="full,none")
    ap.add_argument("--frontends", default="nemo_va,exact")
    ap.add_argument("--pairings", default="fp32_fp32,fp32_int8dec")
    ap.add_argument("--max-symbols", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab(a.model_dir / "vocab.txt")
    clips = [json.loads(l) for l in a.clips.read_text().splitlines() if l.strip()]
    if a.limit:
        clips = clips[:a.limit]
    chunks = [int(x) for x in a.chunks.split(",")]
    resets, frontends, pairings = a.resets.split(","), a.frontends.split(","), a.pairings.split(",")
    graphs = {}
    for cm, pr in itertools.product(chunks, pairings):
        dec = PAIRINGS[pr]
        decp = a.export_dir / dec[8:] if dec.startswith("@export/") else a.model_dir / dec
        graphs[(cm, pr)] = OrtGraphs(a.export_dir / f"encoder_stream_{cm}ms.onnx", decp)
    report = {"config_echo": {k: str(v) for k, v in vars(a).items()}, "clips": [], "summary": {}}
    rows = []
    for c in clips:
        pcm16 = read_pcm16(Path(c["path"]))
        crow = {"clip_id": c["clip_id"], "family": c.get("family"), "label": c.get("label"), "n_samples": int(len(pcm16)),
                "pcm_i16_sha256": hashlib.sha256(pcm16.tobytes()).hexdigest(), "features": {}, "nemo": {}, "arms": {}}
        x = pcm16.astype(np.float32) / 32768.0
        # (b) whole-clip features vs NeMo preprocessor (same for every config)
        nm = np.load(a.nemo_root / f"c{chunks[0]}_{resets[0]}{a.nemo_suffix}" / f"{c['clip_id']}.mel.npy")
        for kind in ("slaney", "htk"):
            m = offline_mel(x, mel=kind)
            n = min(len(m), len(nm))
            d = np.abs(m[:n] - nm[:n])
            crow["features"][f"offline_{kind}"] = {"frames_ours": int(len(m)), "frames_nemo": int(len(nm)),
                                                   "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                                   "p99_abs": float(np.percentile(d, 99))}
        for cm, rs in itertools.product(chunks, resets):
            ref = json.loads((a.nemo_root / f"c{cm}_{rs}{a.nemo_suffix}" / f"{c['clip_id']}.json").read_text())
            crow["nemo"][f"c{cm}_{rs}"] = {"steps": [s["tokens"] for s in ref["steps"]], "tokens": ref["tokens"],
                                          "eou_steps": ref["eou_calls"], "texts": [s["text"] for s in ref["steps"] if s["text"]]}
            chunk = cm * 16
            padded = np.concatenate([x, np.zeros((-len(x)) % chunk, np.float32)])
            vaf = a.nemo_root / f"c{cm}_none{a.nemo_suffix}" / f"{c['clip_id']}.va_feats.npy"
            nva = np.load(vaf) if vaf.exists() else None
            for fe, pr in itertools.product(frontends, pairings):
                arm = ParakeetEOUArm(graphs[(cm, pr)], vocab, chunk_ms=cm, reset=rs, frontend=fe, max_symbols=a.max_symbols)
                r = run_arm(arm, padded, chunk)
                ours = [s["tokens"] for s in r["steps"]]
                theirs = [s["tokens"] for s in ref["steps"]]
                flat_o = [t for s in ours for t in s]
                nstep = min(len(ours), len(theirs))
                fd = next((i for i in range(nstep) if ours[i] != theirs[i]), None if len(ours) == len(theirs) else nstep)
                eou_o = [s["step"] for s in r["steps"] if arm.eou_id in s["tokens"]]
                key = f"c{cm}_{rs}__{fe}__{pr}"
                ent = {"steps_n": len(ours), "tokens": flat_o, "eou_steps": eou_o, "wall_s": r["wall_s"],
                       "token_parity": flat_o == ref["tokens"], "step_parity": fd is None, "first_diff_step": fd,
                       "edit_distance": edit_distance(flat_o, ref["tokens"]), "eou_step_parity": eou_o == ref["eou_calls"]}
                if r["feat_log"] is not None and nva is not None and rs == "none":
                    n = min(len(nva), len(r["feat_log"]))
                    d = np.abs(r["feat_log"][:n] - nva[:n])
                    ent["rolling_feature_buffer_vs_nemo"] = {"steps": int(n), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                                              "p99_abs": float(np.percentile(d, 99))}
                ent["decoder_out_shape"] = list(graphs[(cm, pr)].out_shape or [])
                crow["arms"][key] = ent
                rows.append({"clip_id": c["clip_id"], "config": key, **{k: ent[k] for k in
                             ("token_parity", "step_parity", "eou_step_parity", "edit_distance", "first_diff_step")}})
        report["clips"].append(crow)
        print(c["clip_id"], {k: v["step_parity"] for k, v in crow["arms"].items()}, flush=True)
    keys = sorted({r["config"] for r in rows})
    for k in keys:
        rr = [r for r in rows if r["config"] == k]
        report["summary"][k] = {"n": len(rr), "step_parity": sum(r["step_parity"] for r in rr),
                                "token_parity": sum(r["token_parity"] for r in rr),
                                "eou_step_parity": sum(r["eou_step_parity"] for r in rr),
                                "mean_edit_distance": float(np.mean([r["edit_distance"] for r in rr]))}
    (a.out / "parity_report.json").write_text(json.dumps(report))
    (a.out / "parity_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
