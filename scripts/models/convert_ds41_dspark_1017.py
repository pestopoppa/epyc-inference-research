#!/usr/bin/env python3
"""Convert unsloth's DeepSeek-V4.1-Flash-1017 DSpark GGUF into the port's `deepseek41-dspark` layout.

INF-77 DS41-W6. The original Sept safetensors converter was lost; this replaces it for 1017 and
works purely GGUF -> GGUF with gguf-py (no requantisation, tensor bytes are copied verbatim).

Inputs
  --src     unsloth dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf (arch `dflash`, 78 tensors)
  --target  shard 1 of the 1017 MXFP4-experts target (holds token_embd/output + the tokenizer)
  --ref     the Sept drafter DeepSeek-V4.1-Flash-DSpark.gguf (FORMAT reference only: KV set,
            tensor order, dtypes). No tensor data is taken from it.
  --out     output GGUF

What it does
  * renames the dflash-native DSpark tensors to the names llama-model's DEEPSEEK41_DSPARK branch
    creates (src/models/dflash.cpp):
        fc                -> blk.0.dspark_main_proj
        enc.output_norm   -> blk.0.dspark_main_norm
        markov_w1         -> dspark_markov_embd
        markov_w2         -> dspark_markov_head
        conf_proj         -> dspark_confidence_proj
  * markov tables: BF16 -> F16 (the Sept drafter's dtype). Range-checked (finite, |max| < 65504);
    the only change is F16 subnormal flush of ~1e-5 of entries below ~6e-8 (1017: 347/324 of 33.1M
    entries, max abs err 3e-8, NMSE ~2e-20), asserted NMSE < 1e-12. The loader's BF16 path is
    therefore never exercised.
  * token_embd / output: copied byte-identically from the 1017 target (tied to the backbone).
  * blk.N.exp_probs_b_vl.bias: present in the Sept drafter, absent in unsloth's; the loader
    registers it TENSOR_NOT_REQUIRED and no graph reads it, so it is omitted (reported).
  * KVs: the Sept drafter's `deepseek41-dspark.*` key set, every architectural value cross-checked
    against the unsloth `dflash.*` value (config.json is byte-identical between releases);
    tokenizer.* copied from the 1017 target (add_bos_token=True, the add_bos fix).

Run with an interpreter that has numpy, isolated:  <venv>/bin/python -I convert_ds41_dspark_1017.py ...
"""
from __future__ import annotations

import argparse
import sys

GGUF_PY = "/mnt/raid0/llm/llama.cpp/gguf-py"

RENAMES = {
    "fc.weight": "blk.0.dspark_main_proj.weight",
    "enc.output_norm.weight": "blk.0.dspark_main_norm.weight",
    "markov_w1.weight": "dspark_markov_embd.weight",
    "markov_w2.weight": "dspark_markov_head.weight",
    "conf_proj.weight": "dspark_confidence_proj.weight",
}
FROM_TARGET = ("token_embd.weight", "output.weight")
OPTIONAL_REF_ONLY_SUFFIX = ".exp_probs_b_vl.bias"

A = "deepseek41-dspark"
# ref key (suffix after "deepseek41-dspark.") -> unsloth dflash key, or a callable on the src fields
CROSS = {
    "block_count": "dflash.block_count",
    "embedding_length": "dflash.embedding_length",
    "context_length": "dflash.context_length",
    "attention.head_count": "dflash.attention.head_count",
    "attention.head_count_kv": "dflash.attention.head_count_kv",
    "attention.key_length": "dflash.attention.key_length",
    "attention.value_length": "dflash.attention.value_length",
    "attention.layer_norm_rms_epsilon": "dflash.attention.layer_norm_rms_epsilon",
    "rope.freq_base": "dflash.rope.freq_base",
    "rope.dimension_count": "dflash.rope.dimension_count",
    "attention.sliding_window": "dflash.attention.sliding_window",
    "attention.q_lora_rank": "dflash.attention.q_lora_rank",
    "attention.output_group_count": "dflash.attention.output_group_count",
    "attention.output_lora_rank": "dflash.attention.output_lora_rank",
    "swiglu_clamp_exp": lambda s: _uniform(s["dflash.swiglu_clamp_exp"]),
    "expert_count": "dflash.expert_count",
    "expert_used_count": "dflash.expert_used_count",
    "expert_feed_forward_length": "dflash.expert_feed_forward_length",
    "expert_shared_count": "dflash.expert_shared_count",
    "expert_weights_scale": "dflash.expert_weights_scale",
    "expert_weights_norm": "dflash.expert_weights_norm",
    # unsloth stores the enum; 4 == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS in the port
    "expert_gating_func": lambda s: {4: "sqrtsoftplus"}[s["dflash.expert_gating_func"]],
    "hyper_connection.count": "dflash.hyper_connection.count",
    "hyper_connection.sinkhorn_iterations": "dflash.hyper_connection.sinkhorn_iterations",
    "hyper_connection.epsilon": "dflash.hyper_connection.epsilon",
    "dspark.block_size": "dflash.block_size",
    "dspark.markov_rank": lambda s: None,  # checked against the tensor shape below
    "dspark.noise_token_id": "tokenizer.ggml.mask_token_id",
    "dspark.n_routed_experts": "dflash.expert_count",
    "dspark.num_experts_per_tok": "dflash.expert_used_count",
    "dspark.target_layer_ids": "dflash.target_layers",
    "attention.compress_ratios": "dflash.attention.compress_ratios",
    "dspark.target_arch": lambda s: "deepseek41",
    "dspark.target_block_count": lambda s: None,  # checked against the target below
    "vocab_size": lambda s: len(s["tokenizer.ggml.token_type"]),
}


def _uniform(v):
    if isinstance(v, list):
        assert len(set(v)) == 1, f"non-uniform per-layer value {v}"
        return v[0]
    return v


def _fields(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        out[k] = f.contents()
    return out


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-6 * max(1e-30, abs(float(a)), abs(float(b))) or float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=GGUF_PY)
    args = ap.parse_args()

    sys.path.insert(0, args.gguf_py)
    import numpy as np
    import gguf

    src = gguf.GGUFReader(args.src)
    tgt = gguf.GGUFReader(args.target)
    ref = gguf.GGUFReader(args.ref)
    sf, tf, rf = _fields(src), _fields(tgt), _fields(ref)
    assert sf["general.architecture"] == "dflash", sf["general.architecture"]
    assert rf["general.architecture"] == A
    assert tf["general.architecture"] == "deepseek41", tf["general.architecture"]

    src_t = {t.name: t for t in src.tensors}
    tgt_t = {t.name: t for t in tgt.tensors}

    # ---- KVs -------------------------------------------------------------------------------
    w = gguf.GGUFWriter(args.out, A, use_temp_file=False)
    w.add_name("DeepSeek V4.1 Flash 1017 DSpark drafter")
    w.add_description(
        "Standalone DSpark speculative drafter for deepseek-ai/DeepSeek-V4.1-Flash-1017, converted "
        "GGUF->GGUF from unsloth dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf (tensor bytes verbatim; "
        "markov tables BF16->F16, NMSE<1e-12). token_embd/output are tied to the backbone and copied "
        "verbatim from the 1017 MXFP4-experts target. INF-77 DS41-W6 "
        "(epyc-inference-research scripts/models/convert_ds41_dspark_1017.py).")
    w.add_file_type(int(rf["general.file_type"]))

    markov_rank = int(src_t["markov_w1.weight"].shape[0])
    tgt_blocks = int(tf["deepseek41.num_hidden_layers"])
    report = []
    for k, f in ref.fields.items():
        if k.startswith("GGUF.") or k in ("general.architecture", "general.name", "general.description",
                                           "general.file_type"):
            continue
        if k.startswith("tokenizer."):
            if k not in tgt.fields:
                raise SystemExit(f"target lacks {k}")
            f2 = tgt.fields[k]
            vt = f2.types[0]
            val = f2.contents()
            if vt == gguf.GGUFValueType.ARRAY:
                w.add_key_value(k, val, vt, sub_type=f2.types[-1])
            else:
                w.add_key_value(k, val, vt)
            continue
        assert k.startswith(A + "."), k
        suffix = k[len(A) + 1:]
        ref_val = f.contents()
        how = CROSS.get(suffix)
        if how is None:
            raise SystemExit(f"no cross-check rule for {k}")
        new = how(sf) if callable(how) else sf[how]
        if suffix == "dspark.markov_rank":
            new = markov_rank
        if suffix == "dspark.target_block_count":
            new = tgt_blocks
        if not _close(new, ref_val):
            report.append(f"KV DIFF {k}: ref={ref_val!r} new={new!r}")
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_key_value(k, new, vt, sub_type=f.types[-1])
        else:
            w.add_key_value(k, new, vt)

    # ---- tensors, in the reference order ------------------------------------------------------
    inv = {v: k for k, v in RENAMES.items()}
    plan = []
    for rt in ref.tensors:
        name = rt.name
        if name in FROM_TARGET:
            st = tgt_t[name]
            data = st.data
            ttype = st.tensor_type
        else:
            sname = inv.get(name, name)
            if sname not in src_t:
                if name.endswith(OPTIONAL_REF_ONLY_SUFFIX):
                    report.append(f"OMIT {name} (absent in unsloth src; loader TENSOR_NOT_REQUIRED, unused)")
                    continue
                raise SystemExit(f"src lacks {sname} (for {name})")
            st = src_t[sname]
            data = st.data
            ttype = st.tensor_type
            if ttype == gguf.GGMLQuantizationType.BF16 and rt.tensor_type == gguf.GGMLQuantizationType.F16:
                u16 = np.ascontiguousarray(data).view(np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                assert np.isfinite(f32).all(), f"{sname}: non-finite"
                mx = float(np.abs(f32).max())
                assert mx < 65504.0, f"{sname}: |max| {mx} overflows F16"
                f16 = f32.astype(np.float16)
                err = f16.astype(np.float32) - f32
                n_diff = int(np.count_nonzero(err))
                nmse = float(np.square(err, dtype=np.float64).sum() / np.square(f32, dtype=np.float64).sum())
                assert nmse < 1e-12, f"{sname}: BF16->F16 NMSE {nmse}"
                data = f16.reshape(tuple(int(x) for x in reversed(st.shape)))
                ttype = gguf.GGMLQuantizationType.F16
                report.append(f"CAST {sname} BF16->F16 (|max|={mx:.4g}, {n_diff} entries changed, "
                              f"max|err|={float(np.abs(err).max()):.3g}, NMSE={nmse:.3g})")
        if list(map(int, st.shape)) != list(map(int, rt.shape)) or ttype != rt.tensor_type:
            report.append(f"SHAPE/TYPE DIFF {name}: ref={rt.tensor_type.name}{list(map(int, rt.shape))} "
                          f"new={ttype.name}{list(map(int, st.shape))}")
        plan.append((name, data, ttype))

    used = {inv.get(n, n) for n, _, _ in plan}
    leftovers = sorted(set(src_t) - used)
    if leftovers:
        raise SystemExit(f"unconsumed src tensors: {leftovers}")

    for name, data, ttype in plan:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for name, data, ttype in plan:
        w.write_tensor_data(data)
    w.close()

    for line in report:
        print(line)
    print(f"wrote {args.out}: {len(plan)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
