"""CS-46: Parakeet-Realtime-EOU-120M onnxruntime CPU arm + NeMo parity gate.

Interface for CS-47: ``ParakeetEOUArm(OrtGraphs(encoder, decoder_joint), load_vocab(vocab.txt), chunk_ms=160|80,
reset=none|decoder|full, hold_s=..., frontend=nemo_va|exact).feed(pcm) -> [{"t", "kind", ...}]``.

Reproduce (arm venv: onnxruntime numpy soundfile scipy; NeMo venv: nemo_toolkit[asr]>=2.5.3 + CPU torch):
  1. NeMo venv: python -I export_streaming_models.py --nemo parakeet_realtime_eou_120m-v1.nemo --out EXPORT
       (cache-aware encoders for 160 ms and 80 ms + step-wise fp32 decoder_joint; ~15 s on CPU)
  2. arm venv:  python -I quantize_decoder.py EXPORT/decoder_joint_self_fp32.onnx EXPORT/decoder_joint_self_int8.onnx
  3. NeMo venv: python -I nemo_ref.py --nemo ... --clips CLIPS.jsonl --out REF/c160_none_eval --chunk-ms 160 --reset none --frontend-mode eval
  4. arm venv:  python -I parity.py --clips CLIPS.jsonl --nemo-root REF --nemo-suffix _eval --model-dir COMMUNITY_DIR --export-dir EXPORT --out OUT
  5. inspect_io.py (graph I/O vs the parakeet-rs contract), compare_parakeet_rs.py (secondary comparator), run_clip.py (event stream).
All CPU work runs under a ``region-lock run --role build`` claim (correctness only; no timing claims).
"""
