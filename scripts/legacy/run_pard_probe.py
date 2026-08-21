from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from transformers import AutoTokenizer, TextStreamer

from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    PardSpecDecodeConfig,
)

torch_dtype = torch.bfloat16
kv_cache_type = KVCacheType.DYNAMIC

NUM_SPECULATIVE_TOKENS = 10

model_choice = "qwen" #"deepseek"
print("Model choice model_choice:", model_choice)

BASE = Path("/mnt/raid0/llm/hf")
if model_choice == "qwen":
    TARGET_MODEL_PATH = BASE / "Qwen2.5-7B-Instruct"
    DRAFT_MODEL_PATH  = BASE / "PARD-Qwen2.5-0.5B"
elif model_choice == "deepseek":
    TARGET_MODEL_PATH = BASE / "DeepSeek-R1-Distill-Qwen-32B"
    DRAFT_MODEL_PATH  = BASE / "PARD-DeepSeek-R1-Distill-Qwen-1.5B"
else:
    print("Model selected through model_choice variable is not supported!")

torch.set_num_threads(96)         # physical cores
torch.set_num_interop_threads(1)


def build_inputs(tokenizer: AutoTokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are a very well-versed academic historian.",
        },
        {
            "role": "user",
            "content": "Tell me about the Battle of the Catalaunian Plains."
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer.batch_encode_plus(
        [text],
        return_tensors="pt",
        padding="longest",
    )

    return encoded

def build_model(use_pard: bool):
    pard_config = None
    if use_pard:
        pard_config = PardSpecDecodeConfig(
            model_name_or_path=str(DRAFT_MODEL_PATH),
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        )

    return LLMModel(
        str(TARGET_MODEL_PATH),
        dtype=torch_dtype,
        pard_config=pard_config,
        kv_cache_type=kv_cache_type,
    )

def main():
    model_name = str(TARGET_MODEL_PATH)

    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = build_inputs(tokenizer)

    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": False,
        "temperature": 0.0,
        "random_seed": 123,
    }
    sampling_config = SamplingConfig(**gen_kwargs)

    pace_model = build_model(use_pard=True)

    underlying = pace_model.generator.model
    print("Underlying model dtype:", next(underlying.parameters()).dtype)

    
    text_streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    # Warmup run (optional, not timed)
    print("\nWarmup run (not timed)...")
    _ = pace_model.generate(inputs, sampling_config, text_streamer)
    print("\nWarmup complete.\n")

    # Timed run
    print("Timed run with PARD...\n")
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = pace_model.generate(inputs, sampling_config, text_streamer)
    end = time.perf_counter()

    elapsed = end - start

    # outputs.output_token_ids includes input + generated tokens
    output_ids = outputs.output_token_ids[0]
    input_len = inputs["input_ids"].shape[1]
    total_len = output_ids.shape[0]
    new_tokens = total_len - input_len

    toks_per_sec = new_tokens / elapsed if elapsed > 0 else float("nan")

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Elapsed time : {elapsed:.3f} s")
    print(f"New tokens   : {new_tokens}")
    print(f"Tokens/sec   : {toks_per_sec:.2f}")
    print(f"Input length : {input_len}")
    print(f"Total length : {total_len}")
    print("\nSpeculative stats:", outputs.speculative_stats)


if __name__ == "__main__":
    main()
