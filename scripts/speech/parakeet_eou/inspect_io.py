"""CS-46 (a): metadata-only load of the community Parakeet-EOU ONNX graphs and a
diff of their I/O names / shapes / dtypes / cache layout against the
parakeet-rs @6ad1501 contract (src/model_eou.rs).  No inference is run.

usage: python -I inspect_io.py --model-dir DIR --out report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# parakeet-rs @6ad1501 src/model_eou.rs (ort_encoder / ort_decoder / EncoderCache::new).
# Shapes use None for dynamic dims; ort element types as ONNX strings.
CONTRACT = {
    "encoder": {
        "inputs": {
            "audio_signal": ("float", [1, 128, None]),
            "length": ("int64", [1]),
            "cache_last_channel": ("float", [17, 1, 70, 512]),
            "cache_last_time": ("float", [17, 1, 512, 8]),
            "cache_last_channel_len": ("int64", [1]),
        },
        "outputs": {
            "outputs": ("float", [1, 512, None]),
            "new_cache_last_channel": ("float", [17, 1, 70, 512]),
            "new_cache_last_time": ("float", [17, 1, 512, 8]),
            "new_cache_last_channel_len": ("int64", [1]),
        },
    },
    "decoder_joint": {
        "inputs": {
            "encoder_outputs": ("float", [1, 512, 1]),
            "targets": ("int32", [1, 1]),
            "target_length": ("int32", [1]),
            "input_states_1": ("float", [1, 1, 640]),
            "input_states_2": ("float", [1, 1, 640]),
        },
        "outputs": {
            "outputs": ("float", [1, 1, 1, 1027]),   # parakeet-rs reads l_shape[3] as vocab
            "output_states_1": ("float", [1, 1, 640]),
            "output_states_2": ("float", [1, 1, 640]),
        },
    },
}

_TY = {"tensor(float)": "float", "tensor(int64)": "int64", "tensor(int32)": "int32",
       "tensor(float16)": "float16", "tensor(int8)": "int8", "tensor(uint8)": "uint8"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _norm_shape(shape):
    return [d if isinstance(d, int) else None for d in shape]


def load_meta(path: Path) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    # Creating a session parses + optimises the graph; no run() is ever called.
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def pack(vs):
        return {v.name: {"dtype": _TY.get(v.type, v.type), "shape": _norm_shape(v.shape),
                         "raw_shape": [d if isinstance(d, (int, str)) or d is None else str(d) for d in v.shape]}
                for v in vs}
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
            "inputs": pack(sess.get_inputs()), "outputs": pack(sess.get_outputs()),
            "providers": sess.get_providers()}


def _shape_compat(got, want):
    if len(got) != len(want):
        return False
    return all(g is None or w is None or g == w for g, w in zip(got, want))


def diff(kind: str, meta: dict) -> list[dict]:
    rows = []
    for side in ("inputs", "outputs"):
        want = CONTRACT[kind][side]
        have = meta[side]
        for n, (wt, ws) in want.items():
            if n not in have:
                rows.append({"side": side, "name": n, "status": "MISSING_IN_GRAPH", "contract": [wt, ws]})
                continue
            h = have[n]
            ok_t = h["dtype"] == wt
            ok_s = _shape_compat(h["shape"], ws)
            rows.append({"side": side, "name": n,
                         "status": "MATCH" if ok_t and ok_s else "DIFF",
                         "contract": [wt, ws], "graph": [h["dtype"], h["raw_shape"]]})
        for n in have:
            if n not in want:
                rows.append({"side": side, "name": n, "status": "EXTRA_IN_GRAPH",
                             "graph": [have[n]["dtype"], have[n]["raw_shape"]]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--export-dir", type=Path, default=None, help="also inspect the CS-46 self-exported graphs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    files = {
        "encoder": "encoder-model.onnx",
        "decoder_joint": "decoder_joint-model.onnx",
        "decoder_joint_int8": "decoder_joint-model.int8.onnx",
    }
    rep = {"config_echo": {"model_dir": str(a.model_dir), "files": files,
                           "hf_repo": "ysdede/parakeet-realtime-eou-120m-v1-onnx",
                           "hf_revision": "6d6be8e9113b4aa8ae7b4d5dfb655795c084d0c6",
                           "contract": "parakeet-rs 6ad1501898842fda76f86f7bcad6b361daf5df9e src/model_eou.rs"},
           "graphs": {}, "diff": {}}
    paths = {k: a.model_dir / fn for k, fn in files.items()}
    if a.export_dir:
        for k, fn in (("selfexport_encoder_160ms", "encoder_stream_160ms.onnx"), ("selfexport_encoder_80ms", "encoder_stream_80ms.onnx"),
                      ("selfexport_decoder_joint", "decoder_joint_self_fp32.onnx")):
            paths[k] = a.export_dir / fn
    for k, p in paths.items():
        m = load_meta(p)
        rep["graphs"][k] = m
        rep["diff"][k] = diff("decoder_joint" if "decoder" in k else "encoder", m)
    cfg = json.loads((a.model_dir / "config.json").read_text())
    rep["card_config"] = cfg
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2))
    bad = [(k, r) for k, rows in rep["diff"].items() for r in rows if r["status"] != "MATCH"]
    for k, r in bad:
        print("NON-MATCH", k, json.dumps(r))
    print("graphs:", ", ".join(rep["graphs"]), "| non-matches:", len(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
