# MiniCPM-SALA CPU reference (Z9)

`scripts/benchmark/sala_cpu_oracle.py` supplies the constant-decay Simple GLA recurrence used by the released MiniCPM-SALA source. Inputs are float32 CPU tensors in batch/time/head/feature order. It supports initial/final recurrent state, packed independent sequences, native chunk/fused call signatures and the original ALiBi slope schedule. The native vendor attention method can call it through temporary FLA import modules in an isolated CPU benchmark process; existing FLA imports are refused rather than replaced.

`load_native_sala_cpu(directory, expected_sources)` checks the two local Python-source digests before executing them, imports the vendor class without loading weights, and exposes `CPUForCausalLM` with `GenerationMixin` compatibility. Loading the tokenizer/weights and generating tokens remain explicit caller actions requiring their own resource admission. The CPU-only PyTorch process selects the vendor CPU dense fallback for sparse layers. This module does not implement InfLLM-V2 sparse selection or a GGUF architecture.

Verified source revision: `9180fe1db74f71fb81bc7105efe88f6b19c959b0`. Required source hashes:

| File | SHA-256 |
|---|---|
| configuration_minicpm_sala.py |0bd0668a2c771911c6cc5bf64481a28b23a56aa63dbb60cbc9c75031b6b2cae2|
| modeling_minicpm_sala.py |b6060c08cfd45868f0e90726e48292689b15f01bf9828cf07f6257530477719c|

The controls cover an independently expanded causal sum at lengths0/1/7/63/64/65/129, split continuation, unchanged input state, packed state isolation, invalid metadata/dtype/scale, import cleanup and native source rejection. With `SALA_HF_SOURCE_DIR` set to the pinned source directory, the additional control imports the original vendor module and exercises its fused/short and chunk/65-token attention paths with masked sequences, plus its slope schedule for1/3/7/32 heads. Tested environment: CPU PyTorch2.12.0, Transformers4.57.1 and einops0.8.2; no package installation is performed by the helper.

Run the controls in an admitted model-free CPU prep slot:

```bash
SALA_HF_SOURCE_DIR=/mnt/raid0/llm/models/MiniCPM-SALA-HF-9180fe1d \
  python -m pytest -q tests/test_sala_cpu_oracle.py
```

These are mathematical and native-interface controls, without pretrained-model inference. They provide the Z9 CPU oracle prerequisite. G18 still requires a fresh-production experimental GGUF port and token-identical output on all three original prompts ×32 tokens, each below8192 tokens; neither those outputs nor GPU Triton numerical parity are claimed here. Production kernels and serving registries are unchanged.
