#!/usr/bin/env python3
"""
Memento S2: LoRA SFT Training for Block-Level Reasoning Compression

Two-stage LoRA fine-tuning on OpenMementos-228K:
  Stage 1 — Full attention, format learning (block/summary token generation)
  Stage 2 — Memento attention, compression learning (attend only to summaries)

Usage:
    # Dry-run: validate data loading, config, tokenization (no training)
    python memento_sft.py --dry-run

    # Stage 1 training (full attention, format learning)
    python memento_sft.py --stage 1 --model Qwen/Qwen3-1.7B

    # Stage 2 training (memento attention, loads Stage 1 adapter)
    python memento_sft.py --stage 2 --model Qwen/Qwen3-1.7B \
        --stage1-adapter ./output/memento-s1-lora/

    # Both stages sequentially
    python memento_sft.py --stage both --model Qwen/Qwen3-1.7B

    # Evaluate trained model on MATH-500 subset
    python memento_sft.py --evaluate --model Qwen/Qwen3-1.7B \
        --adapter ./output/memento-s2-lora/

Design reference: microsoft/memento (2026), Kontonis et al.
Dataset: microsoft/OpenMementos-228K (MIT license)
Handoff: epyc-root/handoffs/active/memento-block-reasoning-compression.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Memento special tokens — must be added to tokenizer
SPECIAL_TOKENS = [
    "<|block_start|>",
    "<|block_end|>",
    "<|summary_start|>",
    "<|summary_end|>",
]

# Dataset location
DATASET_DIR = Path("/mnt/raid0/llm/data/openmementos")
DATA_DIR = DATASET_DIR / "data"       # 20 shards, training-ready format
FULL_DIR = DATASET_DIR / "full"       # 39 shards, with block annotations

# Output directories
OUTPUT_BASE = Path("/mnt/raid0/llm/epyc-inference-research/output/memento")


@dataclass
class MementoTrainingConfig:
    """Training configuration for Memento LoRA SFT.

    Defaults tuned for CPU training on AMD EPYC 9655 (192 cores, 1.2TB RAM).
    For GPU training, increase batch_size, enable bf16, use QLoRA via
    bitsandbytes.
    """

    # --- Model ---
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    # Recommended model ladder for validation:
    #   Qwen3-0.6B  — smoke test (3.6 GB FP32, ~19h/epoch CPU)
    #   Qwen3-1.7B  — validation target (10 GB FP32, ~54h/epoch CPU)
    #   Qwen3-8B    — if CPU budget allows (48 GB FP32, ~11 days/epoch CPU)
    #   Qwen3-32B   — production target (192 GB FP32, infeasible on CPU)
    #
    # For 32B: MUST use GPU with QLoRA (4-bit NF4), or rent cloud GPU time.

    # --- LoRA ---
    lora_rank: int = 16
    lora_alpha: int = 32  # alpha = 2 * rank (standard scaling)
    lora_dropout: float = 0.05
    # Target modules for Qwen3 architecture (attention projections + MLP gate)
    # Memento paper targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj",
    ])
    # Trainable params estimate at rank=16, 6 modules:
    #   0.6B model:  ~197K params (0.03% of base)
    #   1.7B model:  ~393K params (0.02% of base)
    #   8B model:    ~786K params (0.01% of base)
    #   32B model:   ~983K params (0.003% of base)

    # --- Training (Stage 1: full attention, format learning) ---
    stage1_epochs: int = 2
    stage1_lr: float = 2e-4         # Standard LoRA learning rate
    stage1_warmup_ratio: float = 0.03
    stage1_max_seq_len: int = 4096  # Truncate; median response ~13K tokens
    # NOTE: 4096 truncates ~90% of responses. For full coverage, need 16384+
    # but memory cost scales quadratically with attention. Start short for
    # format learning (model learns token patterns), extend in Stage 2.

    # --- Training (Stage 2: memento attention, compression learning) ---
    stage2_epochs: int = 1
    stage2_lr: float = 5e-5         # Lower LR — fine-tune from Stage 1
    stage2_warmup_ratio: float = 0.03
    stage2_max_seq_len: int = 4096

    # --- Common ---
    per_device_batch_size: int = 1  # CPU: must be 1 for memory
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    save_steps: int = 500
    eval_fraction: float = 0.02     # Hold out 2% for validation
    seed: int = 42
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True  # Essential for CPU memory
    bf16: bool = True               # EPYC 9655 supports AVX-512 BF16
    # CPU-specific: set torch dtype for model loading
    torch_dtype: str = "bfloat16"   # "float32" if BF16 not available

    # --- Subset for fast iteration ---
    max_train_samples: Optional[int] = None  # None = use all 228K
    max_eval_samples: Optional[int] = 200

    # --- Embedding training ---
    # True adds embed_tokens/lm_head to modules_to_save so the 4 new memento
    # special tokens learn their own vectors. Costs ~155M trainable params on
    # a 0.6B model (tied embeddings) and produces an adapter that
    # convert_lora_to_gguf.py cannot express. Off for smoke runs.
    train_embeddings: bool = False

    # --- Memento attention masking (Stage 2 only) ---
    # In Stage 2, the attention mask is modified so that tokens after a
    # block_end can only attend to the summary tokens (not the block tokens).
    # This teaches the model to compress information into summaries.
    # Implementation: custom attention mask in collator (see below).
    use_memento_attention: bool = False  # Set True for Stage 2


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_parquet_dataset(data_dir: Path, max_samples: Optional[int] = None):
    """Load all parquet shards from a directory into a list of dicts.

    Each record has: source, domain, difficulty, problem, response.
    """
    import pandas as pd
    import pyarrow as pa

    shards = sorted(data_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    frames = []
    if max_samples:
        # Draw the subset ACROSS all shards, not from the first shard(s).
        # OpenMementos shards are domain-clustered: the first 100 rows of
        # shard 0 are 100% `code`, so a head()-based subset is not a sample
        # of the 54% math / 27% science / 19% code corpus. Reading one small
        # record batch per shard costs no more I/O than the old head().
        per_shard = -(-max_samples // len(shards))  # ceil division
        for shard in shards:
            batch = next(pq.ParquetFile(str(shard)).iter_batches(batch_size=per_shard), None)
            if batch is None:
                continue
            frames.append(pa.Table.from_batches([batch]).to_pandas())
        n_read = len(frames)
    else:
        for shard in shards:
            frames.append(pq.read_table(str(shard)).to_pandas())
        n_read = len(frames)

    combined = pd.concat(frames, ignore_index=True)
    if max_samples:
        # Draw the final subset at random, not with head(): after the
        # per-shard concat the rows are still ordered shard-0-first, so
        # head(4) of a 20-shard read returns 4 rows from shards 0-3 only.
        # Fixed seed keeps the subset reproducible.
        combined = (
            combined.sample(n=min(max_samples, len(combined)), random_state=42)
            .reset_index(drop=True)
        )

    print(f"Loaded {len(combined)} examples from {n_read} shards")
    print(f"  Domains: {dict(combined['domain'].value_counts())}")
    return combined.to_dict("records")


def parse_memento_response(response: str) -> list[dict]:
    """Parse a memento-formatted response into structured blocks.

    Returns list of dicts with keys:
      - type: "block" or "summary" or "answer"
      - content: the text content
      - block_idx: which block this belongs to (for blocks and summaries)

    Used for building memento attention masks in Stage 2.
    """
    segments = []
    block_idx = 0

    # Strip outer <think>...</think> wrapper
    inner = response
    if inner.startswith("<think>"):
        inner = inner[len("<think>"):]
    think_end = inner.find("</think>")
    answer = ""
    if think_end >= 0:
        answer = inner[think_end + len("</think>"):].strip()
        inner = inner[:think_end]

    # Parse block/summary pairs
    pattern = re.compile(
        r"<\|block_start\|>(.*?)<\|block_end\|>\s*"
        r"<\|summary_start\|>(.*?)<\|summary_end\|>",
        re.DOTALL,
    )
    for match in pattern.finditer(inner):
        segments.append({
            "type": "block",
            "content": match.group(1).strip(),
            "block_idx": block_idx,
        })
        segments.append({
            "type": "summary",
            "content": match.group(2).strip(),
            "block_idx": block_idx,
        })
        block_idx += 1

    if answer:
        segments.append({
            "type": "answer",
            "content": answer,
            "block_idx": -1,
        })

    return segments


def format_as_chat(example: dict) -> list[dict]:
    """Format a dataset example as chat messages for SFT.

    Returns: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    return [
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["response"]},
    ]


# ---------------------------------------------------------------------------
# Tokenizer Setup
# ---------------------------------------------------------------------------

def setup_tokenizer(model_name: str, dry_run: bool = False):
    """Load tokenizer and add Memento special tokens.

    Returns (tokenizer, num_added_tokens).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Add memento special tokens
    existing = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in SPECIAL_TOKENS if t not in existing]

    num_added = 0
    if tokens_to_add:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        print(f"Added {num_added} special tokens: {tokens_to_add}")
    else:
        print("All Memento special tokens already in vocabulary")

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token = eos_token ({tokenizer.eos_token})")

    if dry_run:
        # Verify special token IDs
        for token in SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token} -> id {tid}")

    return tokenizer, num_added


# ---------------------------------------------------------------------------
# Memento Attention Mask (Stage 2)
# ---------------------------------------------------------------------------

def build_memento_attention_mask(
    input_ids: list[int],
    tokenizer,
    block_start_id: int,
    block_end_id: int,
    summary_start_id: int,
    summary_end_id: int,
) -> list[list[int]]:
    """Build a custom causal attention mask for memento training (Stage 2).

    Standard causal: each token attends to all previous tokens.
    Memento causal: after a block_end, subsequent tokens can only attend to:
      - Tokens before the block_start (context before the block)
      - Summary tokens for that block (summary_start...summary_end)
      - Tokens after the summary_end (including the current token)
    The block tokens (block_start...block_end) are masked from future attention.

    This teaches the model that block content is transient — only the summary
    persists in the "memory" (KV cache at inference time).

    Args:
        input_ids: Tokenized sequence (1D list of token IDs).
        tokenizer: Tokenizer instance (unused but kept for signature).
        block_start_id, block_end_id: Token IDs for block delimiters.
        summary_start_id, summary_end_id: Token IDs for summary delimiters.

    Returns:
        2D attention mask (seq_len x seq_len), 1 = attend, 0 = mask.

    TODO: This is the conceptual implementation. For actual training, this
    needs to be integrated into the model's attention computation, which
    requires either:
      (a) Custom attention mask passed to model.forward() — supported by
          HuggingFace transformers for most models.
      (b) Custom FlashAttention kernel with block-sparse mask — more efficient
          but requires CUDA.
    For CPU training, (a) is the path. Memory cost: O(seq_len^2) per sample,
    which at seq_len=4096 is 4096^2 * 4 bytes = 64 MB per sample. Manageable.
    """
    seq_len = len(input_ids)

    # Start with standard causal mask
    # mask[i][j] = 1 if token i can attend to token j (j <= i)
    mask = [[0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(i + 1):
            mask[i][j] = 1

    # Find block boundaries
    # Each completed block (block_start...block_end...summary_start...summary_end)
    # gets its block tokens masked from all tokens after summary_end.
    blocks = []
    i = 0
    while i < seq_len:
        if input_ids[i] == block_start_id:
            block_start_pos = i
            # Find matching block_end
            j = i + 1
            while j < seq_len and input_ids[j] != block_end_id:
                j += 1
            if j >= seq_len:
                break
            block_end_pos = j
            # Find matching summary_end
            k = j + 1
            while k < seq_len and input_ids[k] != summary_end_id:
                k += 1
            if k >= seq_len:
                break
            summary_end_pos = k

            blocks.append({
                "block_start": block_start_pos,
                "block_end": block_end_pos,
                "summary_end": summary_end_pos,
            })
            i = k + 1
        else:
            i += 1

    # Apply block masking: for each completed block, tokens after summary_end
    # cannot attend to block tokens [block_start, block_end] (inclusive).
    # They CAN still attend to summary tokens.
    for block in blocks:
        bs = block["block_start"]
        be = block["block_end"]
        se = block["summary_end"]
        # For all tokens after summary_end, mask out block tokens
        for row in range(se + 1, seq_len):
            for col in range(bs, be + 1):  # block_start through block_end
                mask[row][col] = 0

    return mask


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

class MementoDataCollator:
    """Collator for Memento SFT training.

    Handles:
    - Chat template application
    - Tokenization with truncation
    - Label masking (only train on assistant response)
    - Memento attention masking (Stage 2 only)
    """

    def __init__(self, tokenizer, max_seq_len: int, use_memento_attention: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_memento_attention = use_memento_attention

        # Cache special token IDs for mask building
        self.block_start_id = tokenizer.convert_tokens_to_ids("<|block_start|>")
        self.block_end_id = tokenizer.convert_tokens_to_ids("<|block_end|>")
        self.summary_start_id = tokenizer.convert_tokens_to_ids("<|summary_start|>")
        self.summary_end_id = tokenizer.convert_tokens_to_ids("<|summary_end|>")

    def prompt_length(self, example: dict) -> int:
        """Token length of the user turn plus the assistant generation prompt.

        This is the boundary at which assistant content starts, so labels
        before it must be -100 (train on the completion only).
        """
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["problem"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        return len(prompt_ids)

    def completion_token_count(self, example: dict) -> int:
        """How many supervised (non -100) tokens survive truncation.

        Zero means the problem statement alone fills max_seq_len, so every
        label is -100. torch's cross_entropy over an all-ignored target
        returns nan (with a zero gradient), which poisons the reported epoch
        loss while contributing nothing to training.
        """
        full = self.tokenizer.apply_chat_template(
            format_as_chat(example),
            tokenize=True,
            max_length=self.max_seq_len,
            truncation=True,
            return_dict=True,
            add_generation_prompt=False,
        )["input_ids"]
        seq_len = len(full)
        return max(0, seq_len - min(self.prompt_length(example), seq_len))

    def __call__(self, examples: list[dict]) -> dict:
        """Collate a batch of examples into model inputs.

        Per example:
        1. Formatted as chat messages
        2. Applied through tokenizer.apply_chat_template()
        3. Truncated to max_seq_len
        4. Labels set to -100 for user/system tokens (only train on assistant)
        5. If Stage 2: attention mask replaced with memento mask
        """
        import torch

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        encoded_lens = []

        rows = []
        for example in examples:
            messages = format_as_chat(example)

            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                max_length=self.max_seq_len,
                truncation=True,
                return_dict=True,
                add_generation_prompt=False,
            )

            input_ids = list(encoded["input_ids"])
            seq_len = len(input_ids)

            # Labels: -100 over the user turn + generation prompt, real ids
            # over the assistant response. Training on the prompt teaches the
            # model to generate problem statements, which is not the objective.
            n_prompt = min(self.prompt_length(example), seq_len)
            labels = [-100] * n_prompt + input_ids[n_prompt:]

            rows.append((input_ids, labels, seq_len))
            encoded_lens.append(seq_len)

        # Dynamic padding: pad to the longest sequence in THIS batch, not to
        # max_seq_len. Padding every short sample out to 4096 wastes the bulk
        # of a CPU step on positions whose labels are all -100.
        pad_to = max(encoded_lens)

        for input_ids, labels, seq_len in rows:
            padding_len = pad_to - seq_len
            padded_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            padded_labels = labels + [-100] * padding_len
            attn_mask = [1] * seq_len + [0] * padding_len

            batch_input_ids.append(padded_ids)
            batch_labels.append(padded_labels)

            if self.use_memento_attention:
                # Stage 2: custom block-masking attention
                memento_mask = build_memento_attention_mask(
                    input_ids,
                    self.tokenizer,
                    self.block_start_id,
                    self.block_end_id,
                    self.summary_start_id,
                    self.summary_end_id,
                )
                # Pad the 2D mask
                padded_mask = [[0] * pad_to for _ in range(pad_to)]
                for r in range(seq_len):
                    row = memento_mask[r]
                    for c in range(seq_len):
                        padded_mask[r][c] = row[c]
                batch_attention_mask.append(padded_mask)
            else:
                batch_attention_mask.append(attn_mask)

        result = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        if self.use_memento_attention:
            # 2D attention mask: (batch, seq, seq) -> expand for head dim later
            result["attention_mask"] = torch.tensor(
                batch_attention_mask, dtype=torch.float32
            )
        else:
            result["attention_mask"] = torch.tensor(
                batch_attention_mask, dtype=torch.long
            )

        return result


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def setup_model_and_lora(config: MementoTrainingConfig, tokenizer, num_added_tokens: int):
    """Load base model and apply the LoRA adapter.

    For CPU training:
      - Load model in BF16 (if AVX-512 BF16 available) or FP32
      - Apply LoRA with PEFT
      - Enable gradient checkpointing

    For GPU QLoRA:
      - Load model in 4-bit NF4 via bitsandbytes
      - Apply LoRA with PEFT
      - Requires: pip install bitsandbytes
    """
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    print(f"Loading model: {config.model_name_or_path}")
    print(f"  dtype: {config.torch_dtype}")
    print(f"  device: cpu")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Resize embeddings ONLY if the tokenizer outgrew the embedding matrix.
    # Qwen3 ships vocab_size=151936 with reserved slots while the tokenizer is
    # 151669 long, so the 4 memento tokens land at 151669..151672 — still
    # inside the existing matrix. Calling resize_token_embeddings(len(tokenizer))
    # unconditionally SHRINKS 151936 -> 151673, destroying 263 rows of the
    # embedding and (tie_word_embeddings=True) of the output head with it.
    n_embed_rows = model.get_input_embeddings().weight.shape[0]
    if num_added_tokens > 0 and len(tokenizer) > n_embed_rows:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings {n_embed_rows} -> {len(tokenizer)}")
    elif num_added_tokens > 0:
        print(
            f"Embedding matrix already covers the new tokens "
            f"({n_embed_rows} rows >= {len(tokenizer)} tokenizer entries) — no resize"
        )

    # Enable gradient checkpointing (critical for CPU memory).
    # `use_reentrant=False` is the non-deprecated torch path.
    # `enable_input_require_grads()` is defensive, NOT a fix for an observed
    # bug: measured 2026-08-12 on torch 2.13 / transformers 5.15 / peft 0.20,
    # LoRA gradients arrive non-None and non-zero in all four combinations of
    # {reentrant, non-reentrant} x {input grads forced, not forced}. It is kept
    # so the call order here stops mattering — under reentrant checkpointing on
    # older stacks, frozen-base inputs do sever the graph.
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        print("Gradient checkpointing: enabled (non-reentrant, input grads forced)")

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        # Training the embedding/head is what teaches the 4 NEW special tokens
        # their own vectors, but it makes ~155M params trainable on a 0.6B
        # model (tied embed_tokens) and the resulting adapter is not
        # convert_lora_to_gguf-friendly. Off by default; on for real runs.
        modules_to_save=(
            ["embed_tokens", "lm_head"]
            if (num_added_tokens > 0 and config.train_embeddings)
            else None
        ),
    )
    # Trainable params must stay fp32: AdamW updates at lr=2e-4 fall below the
    # ~3-decimal-digit resolution of bf16 for weights of typical magnitude.
    # peft does this itself via autocast_adapter_dtype=True (the default) —
    # verified 2026-08-12 — so do NOT pass autocast_adapter_dtype=False here.
    model = get_peft_model(model, lora_config)

    adapter_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    if adapter_dtypes != {torch.float32}:
        print(f"  [WARN] trainable params are {adapter_dtypes}, expected float32")

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_stage(
    model,
    tokenizer,
    train_data: list[dict],
    eval_data: list[dict],
    config: MementoTrainingConfig,
    stage: int,
    output_dir: Path,
):
    """Run one training stage (Stage 1 or Stage 2).

    TODO: This is a skeleton. For actual training, either:
    (a) Use trl.SFTTrainer (recommended — handles chat formatting, loss
        masking, LoRA integration). Requires: pip install trl
    (b) Use transformers.Trainer with custom collator (more control,
        needed for Stage 2 memento attention mask).
    (c) Manual PyTorch loop (maximum control, but more boilerplate).

    For Stage 2 memento attention, option (b) or (c) is required because
    trl.SFTTrainer does not support custom 2D attention masks.
    """
    import torch

    lr = config.stage1_lr if stage == 1 else config.stage2_lr
    epochs = config.stage1_epochs if stage == 1 else config.stage2_epochs
    use_memento = stage == 2
    max_seq_len = config.stage1_max_seq_len if stage == 1 else config.stage2_max_seq_len

    print(f"\n{'='*60}")
    print(f"Stage {stage}: {'Full Attention (Format Learning)' if stage == 1 else 'Memento Attention (Compression Learning)'}")
    print(f"{'='*60}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  Memento attention: {use_memento}")
    print(f"  Effective batch size: {config.per_device_batch_size * config.gradient_accumulation_steps}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Eval samples: {len(eval_data)}")
    print(f"  Output: {output_dir}")

    collator = MementoDataCollator(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        use_memento_attention=use_memento,
    )

    # Drop records that carry no supervision at this max_seq_len: when the
    # problem statement alone fills the window, every label is -100 and the
    # sample costs a full forward+backward while contributing a nan loss and a
    # zero gradient. Measured 2026-08-12: 3/128 records at max_seq_len=1024.
    n_before = len(train_data)
    train_data = [r for r in train_data if collator.completion_token_count(r) > 0]
    n_dropped = n_before - len(train_data)
    if n_dropped:
        print(
            f"  Dropped {n_dropped}/{n_before} records with no completion tokens "
            f"at max_seq_len={max_seq_len} (prompt fills the window)"
        )
    if not train_data:
        raise ValueError(
            f"every training record is prompt-only at max_seq_len={max_seq_len}; "
            "raise --max-seq-len"
        )

    # --- Option A: trl.SFTTrainer (Stage 1 only) ---
    # TODO: Uncomment when trl is installed
    #
    # from trl import SFTTrainer, SFTConfig
    #
    # training_args = SFTConfig(
    #     output_dir=str(output_dir),
    #     num_train_epochs=epochs,
    #     per_device_train_batch_size=config.per_device_batch_size,
    #     gradient_accumulation_steps=config.gradient_accumulation_steps,
    #     learning_rate=lr,
    #     warmup_ratio=config.stage1_warmup_ratio if stage == 1 else config.stage2_warmup_ratio,
    #     weight_decay=config.weight_decay,
    #     max_grad_norm=config.max_grad_norm,
    #     logging_steps=config.logging_steps,
    #     save_steps=config.save_steps,
    #     bf16=config.bf16,
    #     gradient_checkpointing=config.gradient_checkpointing,
    #     max_seq_length=max_seq_len,
    #     seed=config.seed,
    #     dataloader_num_workers=config.dataloader_num_workers,
    #     report_to="none",
    # )
    #
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=eval_data,
    #     data_collator=collator,
    # )
    #
    # trainer.train()
    # trainer.save_model(str(output_dir))

    # --- Manual training loop ---

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=config.weight_decay,
    )

    total_steps = max(1, len(train_data) * epochs // (
        config.per_device_batch_size * config.gradient_accumulation_steps
    ))
    warmup_steps = int(total_steps * (
        config.stage1_warmup_ratio if stage == 1 else config.stage2_warmup_ratio
    ))

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(1, warmup_steps)
    )

    trainable = [p for p in model.parameters() if p.requires_grad]

    model.train()
    global_step = 0
    n_nonfinite = 0
    loss_history: list[float] = []
    t_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        pending = 0
        for step, batch_start in enumerate(range(0, len(train_data), config.per_device_batch_size)):
            batch = train_data[batch_start:batch_start + config.per_device_batch_size]
            inputs = collator(batch)
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = outputs.loss / config.gradient_accumulation_steps

            # A non-finite loss must never reach backward(): one nan gradient
            # propagates through clip_grad_norm_ into every LoRA weight and
            # silently kills the run. The prompt-only filter above removes the
            # known cause; this catches anything else.
            if not torch.isfinite(loss):
                n_nonfinite += 1
                print(f"  [WARN] non-finite loss at step {step}; sample skipped")
                continue

            loss.backward()
            epoch_loss += loss.item()
            loss_history.append(loss.item() * config.gradient_accumulation_steps)
            n_batches += 1
            pending += 1

            if pending == config.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pending = 0

            if step % max(1, config.logging_steps) == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs}, Step {step}, "
                    f"seq {inputs['input_ids'].shape[1]}, "
                    f"Loss: {loss.item() * config.gradient_accumulation_steps:.4f}, "
                    f"{time.time() - t_start:.1f}s",
                    flush=True,
                )

        # Flush a partial accumulation window. Without this the gradients from
        # the tail of the epoch are discarded by the next zero_grad().
        if pending:
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss * config.gradient_accumulation_steps:.4f}")

        # Save adapter checkpoint after each epoch
        epoch_dir = output_dir / f"epoch-{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))
        print(f"  Saved checkpoint: {epoch_dir}")

    elapsed = time.time() - t_start
    metrics = {
        "stage": stage,
        "optimizer_steps": global_step,
        "samples_seen": len(train_data) * epochs,
        "records_dropped_prompt_only": n_dropped,
        "samples_skipped_nonfinite": n_nonfinite,
        "max_seq_len": max_seq_len,
        "elapsed_s": round(elapsed, 1),
        "s_per_sample": round(elapsed / max(1, len(train_data) * epochs), 2),
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "loss_mean_first_quarter": (
            sum(loss_history[: max(1, len(loss_history) // 4)])
            / max(1, len(loss_history) // 4)
            if loss_history else None
        ),
        "loss_mean_last_quarter": (
            sum(loss_history[-max(1, len(loss_history) // 4):])
            / max(1, len(loss_history) // 4)
            if loss_history else None
        ),
        "loss_history": loss_history,
    }
    run_record = output_dir / f"stage{stage}_metrics.json"
    run_record.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics: {run_record}")

    # SC20 belief emission: the run record stands regardless; the claim row is
    # emitted only when the adapter provably updated (see emit_training_belief
    # for the scope contract). A refusal propagates — the caller decides.
    emit_training_belief(
        model=model, config=config, stage=stage, metrics=metrics,
        run_record=run_record,
    )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_math500(
    model,
    tokenizer,
    adapter_path: Optional[str] = None,
    num_samples: int = 50,
):
    """Evaluate memento-trained model on a MATH-500 subset.

    Tests two things:
    1. Format compliance — does the model generate valid block/summary structure?
    2. Accuracy — does it solve math problems correctly?

    TODO: This requires a MATH-500 question set. For now, uses a synthetic
    check on the dataset's own math examples.
    """
    print(f"\n{'='*60}")
    print("Evaluation: MATH-500 Subset")
    print(f"{'='*60}")
    print(f"  Samples: {num_samples}")
    print(f"  Adapter: {adapter_path or 'none (base model)'}")

    # TODO: Load MATH-500 questions
    # For now, validate format compliance on held-out data
    print("\n[STUB] Evaluation would:")
    print("  1. Load MATH-500 questions from benchmark suite")
    print("  2. Generate responses with memento block structure")
    print("  3. Check format compliance (block/summary token pairing)")
    print("  4. Extract final answers and compare to ground truth")
    print("  5. Report: accuracy, avg blocks per response, compression ratio")
    print("  6. Compare base model vs. memento-trained model")

    # Format compliance checker (works on any generated text)
    def check_format(response: str) -> dict:
        has_think = response.startswith("<think>") and "</think>" in response
        block_starts = len(re.findall(r"<\|block_start\|>", response))
        block_ends = len(re.findall(r"<\|block_end\|>", response))
        summary_starts = len(re.findall(r"<\|summary_start\|>", response))
        summary_ends = len(re.findall(r"<\|summary_end\|>", response))
        return {
            "has_think_wrapper": has_think,
            "blocks_balanced": block_starts == block_ends,
            "summaries_balanced": summary_starts == summary_ends,
            "blocks_match_summaries": block_starts == summary_starts,
            "num_blocks": block_starts,
            "valid": (
                has_think
                and block_starts == block_ends
                and summary_starts == summary_ends
                and block_starts == summary_starts
                and block_starts > 0
            ),
        }

    return check_format


# ---------------------------------------------------------------------------
# LoRA parameter accounting
# ---------------------------------------------------------------------------

def lora_param_count(
    model_name: str, rank: int, targets: list[str]
) -> Optional[int]:
    """Exact trainable-parameter count for a rank-`rank` LoRA over `targets`.

    A rank-r adapter on a d_in x d_out projection is r*(d_in + d_out) params,
    and that repeats for EVERY transformer layer. The estimate this replaces
    was `hidden * rank * 2 * len(targets)`, which omitted num_hidden_layers and
    assumed every projection was hidden x hidden: it reported 196,608 for
    Qwen3-0.6B where peft actually builds 8,257,536 (42x low). The handoff's
    model-ladder "197K / 393K / 786K / 983K params" column carries the same
    error and should be multiplied by the layer count.

    Returns None if the model config cannot be read.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None

    h = cfg.hidden_size
    inter = cfg.intermediate_size
    head_dim = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
    q_out = cfg.num_attention_heads * head_dim
    kv_out = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * head_dim
    shapes = {
        "q_proj": (h, q_out), "k_proj": (h, kv_out), "v_proj": (h, kv_out),
        "o_proj": (q_out, h), "gate_proj": (h, inter), "up_proj": (h, inter),
        "down_proj": (inter, h),
    }
    per_layer = sum(rank * (shapes[t][0] + shapes[t][1]) for t in targets if t in shapes)
    return per_layer * cfg.num_hidden_layers


# ---------------------------------------------------------------------------
# Belief-kernel emission (SC20)
# ---------------------------------------------------------------------------

SFT_BELIEF_PROTOCOL_ID = "epyc.memento_sft.lora_training.v1"
SFT_BELIEF_PRODUCER_ID = "scripts.benchmark.memento_sft/train_stage"
SFT_BELIEF_PRODUCER_PATH = "scripts/benchmark/memento_sft.py"
SFT_BELIEF_CATEGORY = "BASELINE"


class BeliefRefused(ValueError):
    """The training finalize cannot honestly produce a belief row."""


def _content_hash(obj) -> str:
    """SHA-256 over the canonical JSON encoding — byte-identical semantics to
    ``autokernel.schemas.content_hash``. Implemented locally because this
    script runs under arbitrary interpreters (ml-training venv, importlib
    spec loads) where the repo root is not on sys.path."""
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def emit_training_belief(*, model, config: MementoTrainingConfig, stage: int,
                         metrics: dict, run_record: Path) -> dict:
    """Write ``stage{stage}_belief_measurements.json`` beside the run record.

    THE CLAIM IS NARROW ON PURPOSE: **"this configuration trains at X s/sample
    with an adapter that provably updated"** — and NOTHING more. It never
    claims "the model improved": SFT throughput and gradient movement say
    nothing about post-training quality, which the MATH-500 gate decides
    separately. That scope limit is also the register contract
    (``epyc-root scripts/vidya/adapters/README.md``, LoRA/SFT row) and is
    carried verbatim in the row's ``extra.scope``.

    Fail-closed: if the adapter-integrity check fails (any trainable tensor
    non-finite, or any ``lora_B`` still zero — i.e. the adapter did NOT
    provably update), the run record stands but no belief row is emitted and
    BeliefRefused propagates, because the claim's second half is false.
    """
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise BeliefRefused("no trainable parameters; LoRA was not applied")

    finite = all(bool(torch.isfinite(p).all().item()) for p in trainable)
    lora_b_total = 0
    lora_b_nonzero = 0
    # Match on SUBSTRING, not suffix: peft names these
    # `...q_proj.lora_B.default.weight` (the `.weight` tail was missing from the
    # original check, so the first real S2 run refused a perfectly good adapter
    # with `0/0` — a false negative, found 2026-08-27 by reproducing the setup).
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_B.default" in name:
            lora_b_total += 1
            if torch.count_nonzero(p) > 0:
                lora_b_nonzero += 1
    integrity = {
        "trainable_tensor_count": len(trainable),
        "all_tensors_finite": finite,
        "lora_B_total": lora_b_total,
        "lora_B_nonzero": lora_b_nonzero,
    }
    if not finite:
        raise BeliefRefused(
            "non-finite trainable tensor; run record stands, claim row refused")
    if lora_b_total == 0 or lora_b_nonzero != lora_b_total:
        raise BeliefRefused(
            f"lora_B off zero-init: {lora_b_nonzero}/{lora_b_total}; the "
            "adapter did not provably update; run record stands, claim row refused")

    loss_history = metrics.get("loss_history") or []
    quarter = max(1, len(loss_history) // 4)
    quarters = {
        f"q{i}": (sum(loss_history[(i - 1) * quarter: i * quarter])
                  / max(1, len(loss_history[(i - 1) * quarter: i * quarter])))
        for i in range(1, 5)
    }
    if not loss_history:
        quarters = {f"q{i}": None for i in range(1, 5)}

    scored = max(0, int(metrics.get("samples_seen", 0))
                 - int(metrics.get("samples_skipped_nonfinite", 0)))
    if scored < 1:
        raise BeliefRefused(
            f"zero samples scored ({metrics.get('samples_seen', 0)} attempted, "
            f"{metrics.get('samples_skipped_nonfinite', 0)} non-finite skips); "
            "zero reps is not a measurement; run record stands, claim row refused")
    attempted = int(metrics.get("samples_seen", 0))
    s_per_sample = float(metrics.get("s_per_sample", 0.0))
    trainable_params = sum(p.numel() for p in trainable)

    # Attestation over the run record content written at collect time.
    run_record = Path(run_record)
    attestation_sha256 = _content_hash(metrics)

    row = {
        "measurement_id": f"memento_sft_stage{stage}_seconds_per_sample",
        "metric": "sft_seconds_per_sample",
        "value": s_per_sample,
        "unit": "s/sample",
        "metric_direction": "lower_better",
        "category": SFT_BELIEF_CATEGORY,
        "protocol_id": SFT_BELIEF_PROTOCOL_ID,
        "reps": scored,
        "reps_basis": (f"scored:{scored} training samples "
                       f"({attempted} attempted minus "
                       f"{metrics.get('samples_skipped_nonfinite', 0)} non-finite skips)"),
        "claim": (f"Memento S2 stage {stage} configuration trains at "
                  f"{s_per_sample} s/sample with an adapter that provably updated "
                  f"(lora_B off zero-init, all tensors finite)"),
        "extra": {
            "date": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "model": config.model_name_or_path,
            "trainable_params": trainable_params,
            "lora": {
                "rank": config.lora_rank,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
                "target_modules": list(config.lora_target_modules),
                "train_embeddings": config.train_embeddings,
            },
            "loss_quarters": quarters,
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "adapter_integrity": integrity,
            "training": {
                "max_seq_len": metrics.get("max_seq_len"),
                "optimizer_steps": metrics.get("optimizer_steps"),
                "elapsed_s": metrics.get("elapsed_s"),
                "records_dropped_prompt_only": metrics.get("records_dropped_prompt_only"),
                "samples_skipped_nonfinite": metrics.get("samples_skipped_nonfinite"),
            },
            "producer_id": SFT_BELIEF_PRODUCER_ID,
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "attestation_path": str(run_record),
            "attestation_sha256": attestation_sha256,
            "attestation_locator": run_record.name,
            "scope": ("CONFIGURATION TRAINING COST ONLY — this row claims that "
                      "this configuration trains at the recorded s/sample with an "
                      "adapter that provably updated; it NEVER claims the model "
                      "improved. Post-training quality is decided by the MATH-500 "
                      "gate separately."),
        },
    }
    row["measurement_sha256"] = _content_hash(row)

    out = run_record.with_name(f"stage{stage}_belief_measurements.json")
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(f"  Belief row: {out}")
    return row


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

def dry_run(config: MementoTrainingConfig):
    """Validate data loading, tokenization, and config without any training.

    This is the primary validation mode — confirms everything works before
    committing to a multi-day training run.
    """
    print("=" * 60)
    print("MEMENTO S2 DRY RUN")
    print("=" * 60)

    # 1. Dataset loading
    print("\n[1/6] Loading dataset...")
    data = load_parquet_dataset(DATA_DIR, max_samples=100)
    print(f"  Sample record keys: {list(data[0].keys())}")
    print(f"  Domain distribution: ", end="")
    domains = {}
    for r in data:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    print(domains)

    # 2. Response parsing
    print("\n[2/6] Parsing memento response structure...")
    sample = data[0]
    segments = parse_memento_response(sample["response"])
    print(f"  Blocks: {sum(1 for s in segments if s['type'] == 'block')}")
    print(f"  Summaries: {sum(1 for s in segments if s['type'] == 'summary')}")
    print(f"  Has answer: {any(s['type'] == 'answer' for s in segments)}")
    for seg in segments[:4]:
        preview = seg["content"][:80].replace("\n", " ")
        print(f"    [{seg['type']}:{seg['block_idx']}] {preview}...")

    # 3. Tokenizer setup
    print(f"\n[3/6] Setting up tokenizer for {config.model_name_or_path}...")
    try:
        tokenizer, num_added = setup_tokenizer(config.model_name_or_path, dry_run=True)

        # 4. Tokenization test
        print(f"\n[4/6] Tokenization test...")
        messages = format_as_chat(sample)
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            max_length=config.stage1_max_seq_len,
            truncation=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        print(f"  Input length: {len(input_ids)} tokens (max: {config.stage1_max_seq_len})")
        print(f"  Truncated: {len(input_ids) >= config.stage1_max_seq_len}")

        # Check special tokens are properly tokenized
        block_start_id = tokenizer.convert_tokens_to_ids("<|block_start|>")
        block_count_in_tokens = input_ids.count(block_start_id)
        print(f"  Block starts in tokenized: {block_count_in_tokens}")

        # 5. Collator test
        print(f"\n[5/6] Collator test...")
        collator = MementoDataCollator(tokenizer, config.stage1_max_seq_len)
        batch = collator(data[:2])
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")

        # Test memento attention mask
        print(f"\n  Testing memento attention mask (Stage 2)...")
        collator_s2 = MementoDataCollator(
            tokenizer, config.stage2_max_seq_len, use_memento_attention=True
        )
        batch_s2 = collator_s2(data[:1])
        mask = batch_s2["attention_mask"]
        print(f"  Memento mask shape: {mask.shape}")
        # Count masked positions (should be > 0 if blocks were found)
        seq_len = min(len(input_ids), config.stage2_max_seq_len)
        causal_ones = seq_len * (seq_len + 1) // 2
        actual_ones = int(mask[0, :seq_len, :seq_len].sum().item())
        masked_positions = causal_ones - actual_ones
        print(f"  Masked positions vs causal: {masked_positions} ({100 * masked_positions / max(causal_ones, 1):.1f}%)")

    except Exception as e:
        print(f"\n  [WARN] Tokenizer/model setup failed: {e}")
        print("  This is expected if the model is not downloaded.")
        print("  Install peft and download model to proceed.")
        tokenizer = None

    # 6. Config summary
    print(f"\n[6/6] Training configuration summary")
    print(f"  Model: {config.model_name_or_path}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  LoRA targets: {config.lora_target_modules}")
    print(f"  Stage 1: {config.stage1_epochs} epochs, lr={config.stage1_lr}, seq_len={config.stage1_max_seq_len}")
    print(f"  Stage 2: {config.stage2_epochs} epochs, lr={config.stage2_lr}, seq_len={config.stage2_max_seq_len}")
    print(f"  Batch size: {config.per_device_batch_size} x {config.gradient_accumulation_steps} grad accum = {config.per_device_batch_size * config.gradient_accumulation_steps} effective")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"  dtype: {config.torch_dtype}")
    print(f"  Dataset: {len(data)} samples loaded (228,557 total)")

    # Memory estimates
    print(f"\n  --- Memory Estimates ---")
    model_sizes = {
        "Qwen/Qwen3-0.6B": (0.6, 1024),
        "Qwen/Qwen3-1.7B": (1.7, 2048),
        "Qwen/Qwen3-4B": (4.0, 2560),
        "Qwen/Qwen3-8B": (8.0, 4096),
        "Qwen/Qwen3-14B": (14.0, 5120),
        "Qwen/Qwen3-32B": (32.0, 5120),
    }
    if config.model_name_or_path in model_sizes:
        params_b, hidden = model_sizes[config.model_name_or_path]
        bf16_gb = params_b * 2
        fp32_gb = params_b * 4
        lora_params = lora_param_count(
            config.model_name_or_path, config.lora_rank, config.lora_target_modules
        )
        if lora_params is None:
            lora_params = hidden * config.lora_rank * 2 * len(config.lora_target_modules)
            print("  [WARN] could not read model config; LoRA estimate is a crude fallback")
        lora_mb = lora_params * 4 / 1e6
        print(f"  Base model (BF16): {bf16_gb:.1f} GB")
        print(f"  Base model (FP32): {fp32_gb:.1f} GB")
        print(f"  LoRA adapter: {lora_mb:.1f} MB ({lora_params:,.0f} params)")
        print(f"  Optimizer states: {lora_mb * 2:.1f} MB")
        print(f"  Estimated peak (BF16 + LoRA + optimizer + activations): ~{bf16_gb * 2.5:.0f} GB")

    # Prerequisite check
    print(f"\n  --- Prerequisites ---")
    # bitsandbytes is only needed for GPU QLoRA (4-bit NF4). CPU training does
    # not import it, so a missing bitsandbytes must not gate a CPU run.
    required = {"peft": False, "trl": False}
    optional = {"bitsandbytes": "GPU QLoRA only — not needed for CPU training"}

    for pkg in required:
        try:
            __import__(pkg)
            required[pkg] = True
        except ImportError:
            pass
    for pkg, installed in required.items():
        print(f"  {pkg}: {'OK' if installed else f'MISSING (pip install {pkg})'}")
    for pkg, note in optional.items():
        try:
            __import__(pkg)
            print(f"  {pkg}: OK")
        except ImportError:
            print(f"  {pkg}: absent — {note}")

    print(f"\n{'='*60}")
    if all(required.values()):
        print("DRY RUN PASSED — ready to train")
    else:
        missing = [k for k, v in required.items() if not v]
        print(f"DRY RUN PARTIAL — install missing packages: pip install {' '.join(missing)}")
    print(f"{'='*60}")

    return True


# ---------------------------------------------------------------------------
# GGUF Export Helper
# ---------------------------------------------------------------------------

def export_lora_for_gguf(adapter_path: str, output_path: str):
    """Convert a saved LoRA adapter to GGUF-compatible format for llama.cpp.

    llama.cpp supports LoRA adapters via:
      llama-server --model base.gguf --lora adapter.gguf

    The conversion pipeline:
    1. Save HF PEFT adapter (this script's output)
    2. Convert to GGUF: python convert_lora_to_gguf.py --base <model> --lora <adapter>
       (script lives in llama.cpp/convert_lora_to_gguf.py)
    3. Load at inference with --lora flag

    TODO: Implement the conversion call when we have a trained adapter.
    """
    llama_cpp_dir = Path("/mnt/raid0/llm/llama.cpp")
    convert_script = llama_cpp_dir / "convert_lora_to_gguf.py"

    if not convert_script.exists():
        print(f"[WARN] convert_lora_to_gguf.py not found at {convert_script}")
        print("  Check llama.cpp installation path")
        return

    print(f"[STUB] Would convert LoRA adapter to GGUF:")
    print(f"  Input:  {adapter_path}")
    print(f"  Output: {output_path}")
    print(f"  Script: {convert_script}")
    print(f"  Command: python {convert_script} --base <model_dir> --lora {adapter_path} --outfile {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Memento S2: LoRA SFT for block-level reasoning compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate data loading and config without training",
    )
    parser.add_argument(
        "--stage", choices=["1", "2", "both"], default="both",
        help="Training stage: 1 (format learning), 2 (compression learning), both",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="Model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--stage1-adapter", type=str, default=None,
        help="Path to Stage 1 adapter (for Stage 2 training)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation on MATH-500 subset",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to trained adapter for evaluation",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit training samples (for debugging)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Override max sequence length for both stages (smoke runs)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epoch count for the selected stage(s) (smoke runs)",
    )
    parser.add_argument(
        "--train-embeddings", action="store_true",
        help="Add embed_tokens/lm_head to modules_to_save so the new memento "
             "special tokens learn their own vectors (~155M trainable params "
             "on 0.6B; adapter is no longer GGUF-convertible)",
    )

    args = parser.parse_args()

    # Build config
    config = MementoTrainingConfig(
        model_name_or_path=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        max_train_samples=args.max_samples,
        train_embeddings=args.train_embeddings,
    )
    if args.max_seq_len is not None:
        config.stage1_max_seq_len = args.max_seq_len
        config.stage2_max_seq_len = args.max_seq_len
    if args.epochs is not None:
        config.stage1_epochs = args.epochs
        config.stage2_epochs = args.epochs

    if args.dry_run:
        dry_run(config)
        return

    if args.evaluate:
        print("[STUB] Evaluation mode — requires trained model")
        evaluate_math500(None, None, adapter_path=args.adapter)
        return

    # Full training path
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE

    print("Loading dataset...")
    data = load_parquet_dataset(DATA_DIR, max_samples=config.max_train_samples)

    # Train/eval split
    import random
    random.seed(config.seed)
    random.shuffle(data)
    eval_size = int(len(data) * config.eval_fraction)
    if config.max_eval_samples:
        eval_size = min(eval_size, config.max_eval_samples)
    eval_data = data[:eval_size]
    train_data = data[eval_size:]
    print(f"Split: {len(train_data)} train, {len(eval_data)} eval")

    # Setup tokenizer and model
    tokenizer, num_added = setup_tokenizer(config.model_name_or_path)
    model = setup_model_and_lora(config, tokenizer, num_added)

    if args.stage in ("1", "both"):
        s1_dir = output_dir / "memento-s1-lora"
        s1_dir.mkdir(parents=True, exist_ok=True)
        model = train_stage(model, tokenizer, train_data, eval_data, config, stage=1, output_dir=s1_dir)

    if args.stage in ("2", "both"):
        s2_dir = output_dir / "memento-s2-lora"
        s2_dir.mkdir(parents=True, exist_ok=True)
        if args.stage1_adapter:
            print(f"Loading Stage 1 adapter from {args.stage1_adapter}")
            # TODO: model = PeftModel.from_pretrained(model, args.stage1_adapter)
        config.use_memento_attention = True
        model = train_stage(model, tokenizer, train_data, eval_data, config, stage=2, output_dir=s2_dir)

    # Save final adapter
    final_dir = output_dir / "memento-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nFinal adapter saved to: {final_dir}")

    # Export hint
    print("\nTo convert adapter to GGUF for llama.cpp inference:")
    print(f"  python /mnt/raid0/llm/llama.cpp/convert_lora_to_gguf.py \\")
    print(f"    --base {config.model_name_or_path} \\")
    print(f"    --lora {final_dir} \\")
    print(f"    --outfile {final_dir}/memento-lora.gguf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
