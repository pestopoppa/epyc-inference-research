"""CS-46 self-export of the cache-aware streaming encoder (and fp32 decoder_joint) from the .nemo.

Why: the community graphs (ysdede/parakeet-realtime-eou-120m-v1-onnx @6d6be8e) carry a STATELESS
encoder (inputs audio_signal,length only - no cache tensors), so the cache-aware chunked loop
cannot run on them; see io_contract_diff.json.  The export follows parakeet-rs @6ad1501
``scripts/export_realtime_eou_120m.py`` (encoder.cache_aware_stream_step wrapper,
keep_all_outputs=False, drop_extra_pre_encoded from streaming_cfg), but

  * traces with a realistic 25-frame slice (9 pre-encode + 16),
  * exports TWO encoders: shift 2 (160 ms: valid_out_len 2) and shift 1 (80 ms voice-agent mode:
    valid_out_len 1, cache_drop_size 1), because both constants are baked into the graph,
  * opset 17, dynamic batch/time axes.

Run in the isolated NeMo CPU venv:  python -I export_streaming_models.py --nemo F.nemo --out DIR
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    import nemo.collections.asr as nemo_asr
    logging.getLogger("nemo_logging").setLevel(logging.ERROR)
    try:
        from nemo.core.classes.common import typecheck
        typecheck.set_typecheck_enabled(False)
    except ImportError:
        pass
    a.out.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.restore_from(a.nemo, map_location=torch.device("cpu"))
    model.eval()
    report = {"nemo_file": a.nemo, "torch": torch.__version__, "exports": {}}

    class Wrapper(torch.nn.Module):
        def __init__(self, enc, drop):
            super().__init__()
            self.enc, self.drop = enc, drop

        def forward(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
            return self.enc.cache_aware_stream_step(
                processed_signal=audio_signal, processed_signal_length=length,
                cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len, keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop)

    i_names = ["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"]
    o_names = ["outputs", "encoded_lengths", "new_cache_last_channel", "new_cache_last_time", "new_cache_last_channel_len"]
    for shift, tag in ((2, "160"), (1, "80")):
        model.encoder.setup_streaming_params(chunk_size=2, shift_size=shift)
        cfg = model.encoder.streaming_cfg
        model.encoder.streaming = True
        drop = cfg.drop_extra_pre_encoded
        feats = torch.randn(1, 128, 25)
        flen = torch.tensor([25])
        cache = model.encoder.get_initial_cache_state(batch_size=1)
        w = Wrapper(model.encoder, drop)
        path = a.out / f"encoder_stream_{tag}ms.onnx"
        kw = dict(input_names=i_names, output_names=o_names, opset_version=17,
                  dynamic_axes={"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"},
                                "outputs": {0: "batch", 2: "time"}})
        how = "torchscript(dynamo=False)"
        try:
            torch.onnx.export(w, (feats, flen, *cache), str(path), dynamo=False, **kw)
        except TypeError:
            how = "default"
            torch.onnx.export(w, (feats, flen, *cache), str(path), **kw)
        with torch.no_grad():
            ref = w(feats, flen, *cache)
        report["exports"][tag] = {"path": str(path), "how": how, "streaming_cfg": str(cfg), "drop_extra": int(drop),
                                  "torch_out_frames": int(ref[0].shape[2])}
        print("exported", path, report["exports"][tag], flush=True)

    # fp32 decoder_joint (stateless export path of NeMo; the community decoder is the primary one)
    tmp = a.out / "_full_export"
    tmp.mkdir(exist_ok=True)
    model.export(output=str(tmp / "parakeet_eou.onnx"), check_trace=False)
    for f in tmp.glob("*.onnx"):
        if "decoder" in f.name.lower() and "joint" in f.name.lower():
            f.rename(a.out / "decoder_joint_self_fp32.onnx")
            report["exports"]["decoder_joint_self_fp32"] = str(a.out / "decoder_joint_self_fp32.onnx")
    report["tmp_files"] = sorted(p.name for p in tmp.iterdir())
    (a.out / "export_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
