#!/usr/bin/env python3
"""Regression tests for scripts/benchmark/memento_sft.py (Memento S2 LoRA SFT).

Each test pins a defect found while running the first real Stage-1 smoke on
2026-08-12; the docstring names the mutation that makes it fail again.

Run:
    HF_HOME=/mnt/raid0/llm/hf-home HF_HUB_OFFLINE=1 \
    /mnt/raid0/llm/venvs/ml-training/bin/python -m pytest \
        scripts/benchmark/test_memento_sft.py -v

Model-loading tests are skipped (not silently passed) when the Qwen3-0.6B
snapshot or the training deps are unavailable.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("memento_sft.py")
MODEL = "Qwen/Qwen3-0.6B"


def _load_module():
    spec = importlib.util.spec_from_file_location("memento_sft", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__], so the
    # module must be registered BEFORE exec_module runs.
    sys.modules["memento_sft"] = module
    spec.loader.exec_module(module)
    return module


msft = _load_module()


def _require(*mods):
    for name in mods:
        pytest.importorskip(name)


@pytest.fixture(scope="module")
def tokenizer():
    _require("transformers", "torch")
    tok, num_added = msft.setup_tokenizer(MODEL)
    assert num_added == 4, "the 4 memento special tokens should be new to Qwen3"
    return tok


@pytest.fixture(scope="module")
def sample():
    if not msft.DATA_DIR.exists():
        pytest.skip(f"OpenMementos not staged at {msft.DATA_DIR}")
    return msft.load_parquet_dataset(msft.DATA_DIR, max_samples=8)[0]


# --------------------------------------------------------------------------
# Dataset subsetting
# --------------------------------------------------------------------------

def test_subset_is_drawn_across_shards_not_from_shard_zero():
    """MUTATION: restore `combined.head(max_samples)` over a shard-0-only read.

    OpenMementos shards are domain-clustered — the first 100 rows of shard 0
    are 100% `code`. A head()-based subset therefore trains the smoke on a
    single domain while claiming to sample a 54/27/19 math/science/code corpus.
    """
    if not msft.DATA_DIR.exists():
        pytest.skip(f"OpenMementos not staged at {msft.DATA_DIR}")
    records = msft.load_parquet_dataset(msft.DATA_DIR, max_samples=100)
    domains = {r["domain"] for r in records}
    assert len(records) == 100
    assert len(domains) >= 3, f"subset collapsed to {domains}; shard spread lost"


def test_small_subset_is_still_multi_domain():
    """MUTATION: drop the random `.sample(random_state=42)` final draw.

    Per-shard reads alone are not enough: after the concat the rows are still
    ordered shard-0-first, so head(4) of a 20-shard read returns shards 0-3,
    which are all `code`. Measured 2026-08-12 at n=4: head -> {code: 4},
    random draw -> {code: 2, science: 2}. n=4 is the smallest n that
    discriminates, so this test uses it deliberately.
    """
    if not msft.DATA_DIR.exists():
        pytest.skip(f"OpenMementos not staged at {msft.DATA_DIR}")
    records = msft.load_parquet_dataset(msft.DATA_DIR, max_samples=4)
    assert len(records) == 4
    assert len({r["domain"] for r in records}) >= 2, (
        "4-sample draw collapsed to one domain — final random draw lost"
    )


def test_subset_draw_is_reproducible():
    """The fixed seed must make the smoke slice replayable."""
    if not msft.DATA_DIR.exists():
        pytest.skip(f"OpenMementos not staged at {msft.DATA_DIR}")
    a = msft.load_parquet_dataset(msft.DATA_DIR, max_samples=16)
    b = msft.load_parquet_dataset(msft.DATA_DIR, max_samples=16)
    assert [r["problem"] for r in a] == [r["problem"] for r in b]


# --------------------------------------------------------------------------
# Collator: label masking and padding
# --------------------------------------------------------------------------

def test_labels_mask_the_prompt(tokenizer, sample):
    """MUTATION: restore `labels = list(input_ids)`.

    Training on the user turn teaches the model to generate problem
    statements. Labels must be -100 up to the assistant generation prompt.
    """
    collator = msft.MementoDataCollator(tokenizer, max_seq_len=1024)
    batch = collator([sample])
    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()

    n_prompt = collator.prompt_length(sample)
    assert n_prompt > 0
    assert all(x == -100 for x in labels[:n_prompt]), "prompt span is not masked"
    # The completion span must survive: at least some real label ids remain.
    completion = [x for x in labels[n_prompt:] if x != -100]
    assert len(completion) > 0, "completion span was fully masked — nothing to train on"
    assert completion == input_ids[n_prompt:n_prompt + len(completion)]


def test_dynamic_padding_does_not_pad_to_max_seq_len(tokenizer):
    """MUTATION: restore `padding_len = self.max_seq_len - seq_len`.

    Padding a short example out to 4096 spends the whole CPU step on
    positions whose labels are all -100.
    """
    tiny = {"problem": "2+2?", "response": "<think>four</think>4"}
    collator = msft.MementoDataCollator(tokenizer, max_seq_len=4096)
    batch = collator([tiny])
    assert batch["input_ids"].shape[1] < 100, (
        f"short sample padded to {batch['input_ids'].shape[1]}; dynamic padding lost"
    )


def test_batch_pads_to_longest_member(tokenizer):
    """Mixed-length batches must align on the longest member, not on max_seq_len."""
    short = {"problem": "2+2?", "response": "<think>four</think>4"}
    longer = {"problem": "2+2? " + "explain carefully. " * 40,
              "response": "<think>" + "reason. " * 60 + "</think>4"}
    collator = msft.MementoDataCollator(tokenizer, max_seq_len=4096)
    batch = collator([short, longer])
    width = batch["input_ids"].shape[1]
    assert width == collator.prompt_length(longer) or width > 100
    assert width < 4096
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_prompt_only_record_reports_zero_completion_tokens(tokenizer):
    """MUTATION: make `completion_token_count` return a constant 1.

    When the problem statement alone fills max_seq_len, truncation leaves no
    assistant tokens, every label is -100, and torch's cross_entropy over an
    all-ignored target returns nan. Observed live on 2026-08-12: 3 of 128
    sampled OpenMementos records at max_seq_len=1024, which turned the whole
    Stage-1 epoch's reported average loss into nan.
    """
    long_prompt = {"problem": "solve this. " * 2000, "response": "<think>x</think>7"}
    normal = {"problem": "what is 2+2?", "response": "<think>four</think>4"}
    collator = msft.MementoDataCollator(tokenizer, max_seq_len=256)

    assert collator.completion_token_count(long_prompt) == 0
    assert collator.completion_token_count(normal) > 0


def test_all_masked_batch_really_produces_nan(tokenizer):
    """The premise of the filter: prove an all-ignored target IS nan.

    Without this the filter could be guarding a condition that never hurts.
    """
    import torch

    long_prompt = {"problem": "solve this. " * 2000, "response": "<think>x</think>7"}
    collator = msft.MementoDataCollator(tokenizer, max_seq_len=256)
    batch = collator([long_prompt])
    labels = batch["labels"][0]
    assert (labels == -100).all(), "fixture no longer produces an all-masked batch"

    logits = torch.randn(labels.shape[0], 32, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
    assert torch.isnan(loss), "all-ignored cross_entropy is no longer nan"
    loss.backward()
    assert logits.grad.abs().sum().item() == 0.0, "nan loss now carries gradient"


def test_truncation_still_respects_max_seq_len(tokenizer, sample):
    """A long real record must be capped, not overflow the configured window."""
    collator = msft.MementoDataCollator(tokenizer, max_seq_len=512)
    batch = collator([sample])
    assert batch["input_ids"].shape[1] <= 512


# --------------------------------------------------------------------------
# LoRA parameter accounting
# --------------------------------------------------------------------------

def test_lora_param_estimate_matches_peft():
    """MUTATION: make `lora_param_count` return `h * rank * 2 * len(targets)`.

    That formula omits num_hidden_layers and assumes every projection is
    hidden x hidden. For Qwen3-0.6B it reports 196,608 where peft builds
    8,257,536 — 42x low. The same error produced the handoff's model-ladder
    "197K / 393K params" column.

    The ground truth here is peft's own count, not a second hand-derivation:
    an independent recomputation would pass even if the function under test
    were never called.
    """
    _require("transformers", "peft", "torch")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    rank = 16

    estimate = msft.lora_param_count(MODEL, rank, targets)
    assert estimate is not None, "could not read model config"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    peft_model = get_peft_model(
        base,
        LoraConfig(r=rank, lora_alpha=32, target_modules=targets,
                   task_type=TaskType.CAUSAL_LM, bias="none"),
    )
    actual = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    assert estimate == actual, f"estimate {estimate} != peft's {actual}"
    assert actual == 8_257_536

    # Guard against the test passing because both sides are the naive formula.
    naive = 1024 * rank * 2 * len(targets)
    assert naive == 196_608
    assert actual != naive, "test is vacuous — real count equals the naive one"


# --------------------------------------------------------------------------
# Model setup: resize guard, LoRA application, gradient flow
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lora_model(tokenizer):
    _require("torch", "peft")
    cfg = msft.MementoTrainingConfig(
        model_name_or_path=MODEL,
        gradient_checkpointing=True,
        torch_dtype="bfloat16",
    )
    return msft.setup_model_and_lora(cfg, tokenizer, num_added_tokens=4)


def test_embeddings_are_not_shrunk_by_resize(lora_model, tokenizer):
    """MUTATION: restore the unconditional `resize_token_embeddings(len(tokenizer))`.

    Qwen3 ships vocab_size=151936 with reserved slots while the tokenizer is
    151669 long, so the memento tokens land inside the existing matrix.
    Resizing unconditionally SHRINKS 151936 -> 151673, discarding 263 rows of
    the embedding and, under tie_word_embeddings, of the output head.
    """
    rows = lora_model.get_input_embeddings().weight.shape[0]
    assert len(tokenizer) == 151_673
    assert rows == 151_936, f"embedding matrix is {rows} rows; resize shrank it"


def test_lora_is_applied_and_base_weights_are_frozen(lora_model):
    """MUTATION: comment out `model = get_peft_model(model, lora_config)`.

    Without it every base parameter keeps requires_grad=True and the optimizer
    silently runs a FULL fine-tune while the script reports "LoRA".
    """
    from peft import PeftModel

    assert isinstance(lora_model, PeftModel), "model was never wrapped by peft"

    trainable = [n for n, p in lora_model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in lora_model.named_parameters() if not p.requires_grad]
    assert trainable, "nothing is trainable"
    assert frozen, "everything is trainable — this is a full fine-tune"
    assert all("lora_" in n for n in trainable), f"non-LoRA params trainable: {trainable[:5]}"

    n_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    assert n_trainable == 8_257_536


def test_trainable_params_are_fp32(lora_model):
    """MUTATION: pass `autocast_adapter_dtype=False` to get_peft_model.

    The base model loads in bf16. AdamW updates at lr=2e-4 fall below bf16's
    ~3-decimal-digit resolution for weights of typical magnitude, so bf16 LoRA
    params silently absorb much of the update. peft's default
    autocast_adapter_dtype=True is what keeps the adapter in fp32 — verified
    2026-08-12 — and this test is what stops someone turning it off.
    """
    import torch

    dtypes = {p.dtype for p in lora_model.parameters() if p.requires_grad}
    assert dtypes == {torch.float32}, f"trainable dtypes are {dtypes}"


def test_gradients_reach_lora_under_gradient_checkpointing(lora_model, tokenizer, sample):
    """End-to-end proof that a step of this pipeline actually updates the LoRA.

    MUTATION: mask every label (`labels = [-100] * len(input_ids)` in the
    collator) — a plausible off-by-one in the prompt-masking fix. The loss goes
    nan and every gradient dies, which no loss-curve inspection would catch.

    NOTE: this test does NOT guard `enable_input_require_grads()`. Measured
    2026-08-12 on torch 2.13 / transformers 5.15 / peft 0.20, LoRA gradients
    arrive non-None and non-zero with and without it, reentrant or not. The
    call is kept as ordering-hardening, not as a fix for a live bug.
    """
    import torch

    collator = msft.MementoDataCollator(tokenizer, max_seq_len=512)
    batch = collator([sample])

    lora_model.train()
    lora_model.zero_grad(set_to_none=True)
    out = lora_model(**batch)
    assert torch.isfinite(out.loss), f"loss is {out.loss}"
    out.loss.backward()

    lora_b = {
        n: p for n, p in lora_model.named_parameters()
        if p.requires_grad and "lora_B" in n
    }
    assert lora_b, "no lora_B parameters found"

    missing = [n for n, p in lora_b.items() if p.grad is None]
    assert not missing, f"{len(missing)}/{len(lora_b)} lora_B params got NO gradient"

    nonzero = [n for n, p in lora_b.items() if p.grad.abs().sum().item() > 0]
    assert nonzero, "every lora_B gradient is exactly zero — nothing would update"

    lora_model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
